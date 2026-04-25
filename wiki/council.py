"""
LLM Council for the wiki pipeline.

A structured ensemble of 4 specialist models + 1 chairman model.
Each specialist analyzes source content from a different angle;
the chairman (opus-4.6) reconciles all outputs into final wiki pages.

Call sequence is sequential (not parallel) because later models
benefit from earlier outputs.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from wiki.prompts import (
    CHAIRMAN_CREATE_PAGE_SYSTEM,
    CHAIRMAN_CREATE_PAGE_USER,
    CHAIRMAN_MIGRATION_USER,
    CHAIRMAN_UPDATE_PAGE_SYSTEM,
    CHAIRMAN_UPDATE_PAGE_USER,
    CONTRADICTION_FINDER_SYSTEM,
    CONTRADICTION_FINDER_USER,
    CROSS_CONNECTOR_SYSTEM,
    CROSS_CONNECTOR_USER,
    ENTITY_EXTRACTOR_EXISTING_CONTEXT,
    ENTITY_EXTRACTOR_SYSTEM,
    ENTITY_EXTRACTOR_USER,
    PROSE_SYNTHESIZER_SYSTEM,
    PROSE_SYNTHESIZER_USER,
)
from wiki.rate_limiter import RateLimiter, RateLimitTimeout

logger = logging.getLogger(__name__)


class CouncilRole(str, Enum):
    ENTITY_EXTRACTOR = "entity_extractor"
    PROSE_SYNTHESIZER = "prose_synthesizer"
    CROSS_CONNECTOR = "cross_connector"
    CONTRADICTION_FINDER = "contradiction_finder"
    CHAIRMAN = "chairman"


@dataclass(frozen=True)
class CouncilMember:
    """Configuration for a single council member."""

    role: CouncilRole
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class CouncilOutput:
    """Result from a single council member call."""

    role: CouncilRole
    model_name: str
    content: str = ""
    parsed: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0
    failed: bool = False
    error: str = ""


@dataclass
class CouncilSession:
    """Complete record of a council convening."""

    source_content: str
    source_metadata: Dict[str, str]
    outputs: List[CouncilOutput] = field(default_factory=list)
    chairman_output: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    affected_entities: List[str] = field(default_factory=list)
    council_mode: str = "full"

    @property
    def successful_outputs(self) -> List[CouncilOutput]:
        return [o for o in self.outputs if not o.failed]

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.outputs if o.failed)

    def get_output(self, role: CouncilRole) -> Optional[CouncilOutput]:
        for o in self.outputs:
            if o.role == role:
                return o
        return None


class LLMCouncil:
    """
    Multi-model council for wiki content generation.

    Orchestrates 4 specialist models sequentially, then a chairman
    to reconcile all outputs. Handles rate limiting and graceful
    degradation when individual models fail.
    """

    def __init__(
        self,
        entity_extractor_model: str,
        prose_synthesizer_model: str,
        cross_connector_model: str,
        contradiction_finder_model: str,
        chairman_model: str,
        rate_limiter: RateLimiter,
        provider: str = "litellm",
    ) -> None:
        self.provider = provider
        self.rate_limiter = rate_limiter

        self.members = {
            CouncilRole.ENTITY_EXTRACTOR: CouncilMember(
                role=CouncilRole.ENTITY_EXTRACTOR,
                model_name=entity_extractor_model,
                temperature=0.1,  # Low temp for structured extraction
                max_tokens=4096,
            ),
            CouncilRole.PROSE_SYNTHESIZER: CouncilMember(
                role=CouncilRole.PROSE_SYNTHESIZER,
                model_name=prose_synthesizer_model,
                temperature=0.5,  # Moderate for creative synthesis
                max_tokens=4096,
            ),
            CouncilRole.CROSS_CONNECTOR: CouncilMember(
                role=CouncilRole.CROSS_CONNECTOR,
                model_name=cross_connector_model,
                temperature=0.7,  # Higher for creative connections
                max_tokens=2048,
            ),
            CouncilRole.CONTRADICTION_FINDER: CouncilMember(
                role=CouncilRole.CONTRADICTION_FINDER,
                model_name=contradiction_finder_model,
                temperature=0.2,  # Low for analytical precision
                max_tokens=4096,
            ),
            CouncilRole.CHAIRMAN: CouncilMember(
                role=CouncilRole.CHAIRMAN,
                model_name=chairman_model,
                temperature=0.3,
                max_tokens=8192,  # Chairman needs more space
            ),
        }

    def convene(
        self,
        source_content: str,
        source_metadata: Dict[str, str],
        existing_entities: List[str],
        existing_page_summaries: str = "",
        council_mode: str = "full",
    ) -> CouncilSession:
        """
        Run the full council pipeline for a source.

        council_mode:
          "full" - all 4 specialists + chairman (live ingestion)
          "migration" - entity_extractor + chairman only (bulk migration)
        """
        session = CouncilSession(
            source_content=source_content,
            source_metadata=source_metadata,
            council_mode=council_mode,
        )

        # Step 1: Entity Extractor (always runs)
        entity_output = self._call_member(
            CouncilRole.ENTITY_EXTRACTOR,
            source_content,
            source_metadata,
            existing_entities=existing_entities,
        )
        session.outputs.append(entity_output)

        # Parse entity extractor JSON
        if not entity_output.failed:
            entity_output.parsed = _safe_parse_json(entity_output.content)

        if council_mode == "full":
            # Step 2: Prose Synthesizer (source text only, no contamination)
            prose_output = self._call_member(
                CouncilRole.PROSE_SYNTHESIZER,
                source_content,
                source_metadata,
            )
            session.outputs.append(prose_output)

            # Step 3: Cross Connector (source + entities from step 1)
            entity_list = ""
            if entity_output.parsed:
                entities = entity_output.parsed.get("entities", [])
                entity_list = "\n".join(
                    f"- {e.get('name', '')} ({e.get('type', '')})" for e in entities
                )
            cross_output = self._call_member(
                CouncilRole.CROSS_CONNECTOR,
                source_content,
                source_metadata,
                entity_list=entity_list,
            )
            session.outputs.append(cross_output)

            # Step 4: Contradiction Finder (source + existing pages)
            contra_output = self._call_member(
                CouncilRole.CONTRADICTION_FINDER,
                source_content,
                source_metadata,
                existing_page_summaries=existing_page_summaries,
            )
            session.outputs.append(contra_output)

        return session

    def call_chairman_create(
        self,
        session: CouncilSession,
        entity_name: str,
        entity_type: str,
        display_name: str,
        category: str,
        tags: List[str],
        people: List[str],
        source_count: int,
        confidence: str,
        canonical_entities: List[str],
        today: str = "",
    ) -> str:
        """Ask the chairman to create a new wiki page from council outputs."""
        from datetime import date as date_type

        today = today or date_type.today().isoformat()

        member = self.members[CouncilRole.CHAIRMAN]
        user_msg = CHAIRMAN_CREATE_PAGE_USER.format(
            entity_name=entity_name,
            entity_type=entity_type,
            display_name=display_name,
            category=category,
            tags=tags,
            people_list=people,
            source_count=source_count,
            confidence=confidence,
            date=today,
            entity_extractor_output=_get_output_text(session, CouncilRole.ENTITY_EXTRACTOR),
            prose_synthesizer_output=_get_output_text(session, CouncilRole.PROSE_SYNTHESIZER),
            cross_connector_output=_get_output_text(session, CouncilRole.CROSS_CONNECTOR),
            contradiction_finder_output=_get_output_text(session, CouncilRole.CONTRADICTION_FINDER),
            canonical_entities=", ".join(canonical_entities),
        )

        content = self._llm_call(member, CHAIRMAN_CREATE_PAGE_SYSTEM, user_msg)
        session.chairman_output = content
        return content

    def call_chairman_update(
        self,
        session: CouncilSession,
        existing_page_content: str,
        source_title: str,
        source_ref: str,
        source_slug: str,
        canonical_entities: List[str],
        today: str = "",
    ) -> str:
        """Ask the chairman to produce an XML diff for updating an existing page."""
        from datetime import date as date_type

        today = today or date_type.today().isoformat()

        member = self.members[CouncilRole.CHAIRMAN]
        user_msg = CHAIRMAN_UPDATE_PAGE_USER.format(
            existing_page_content=existing_page_content,
            source_title=source_title,
            source_ref=source_ref,
            source_slug=source_slug,
            date=today,
            entity_extractor_output=_get_output_text(session, CouncilRole.ENTITY_EXTRACTOR),
            prose_synthesizer_output=_get_output_text(session, CouncilRole.PROSE_SYNTHESIZER),
            cross_connector_output=_get_output_text(session, CouncilRole.CROSS_CONNECTOR),
            contradiction_finder_output=_get_output_text(session, CouncilRole.CONTRADICTION_FINDER),
            canonical_entities=", ".join(canonical_entities),
        )

        content = self._llm_call(member, CHAIRMAN_UPDATE_PAGE_SYSTEM, user_msg)
        session.chairman_output = content
        return content

    def call_chairman_migration(
        self,
        session: CouncilSession,
        lessons_text: str,
        category: str,
        batch_size: int,
        canonical_entities: List[str],
        today: str = "",
    ) -> str:
        """Ask the chairman to create wiki pages from a batch of migrated lessons."""
        from datetime import date as date_type

        today = today or date_type.today().isoformat()

        member = self.members[CouncilRole.CHAIRMAN]
        user_msg = CHAIRMAN_MIGRATION_USER.format(
            batch_size=batch_size,
            category=category,
            lessons_text=lessons_text,
            canonical_entities=", ".join(canonical_entities),
            date=today,
        )

        content = self._llm_call(member, CHAIRMAN_CREATE_PAGE_SYSTEM, user_msg)
        session.chairman_output = content
        return content

    def _call_member(
        self,
        role: CouncilRole,
        source_content: str,
        source_metadata: Dict[str, str],
        existing_entities: Optional[List[str]] = None,
        entity_list: str = "",
        existing_page_summaries: str = "",
    ) -> CouncilOutput:
        """Call a single council member with appropriate prompts."""
        member = self.members[role]
        start = time.monotonic()

        try:
            system_msg, user_msg = self._build_prompts(
                role, source_content, source_metadata,
                existing_entities=existing_entities,
                entity_list=entity_list,
                existing_page_summaries=existing_page_summaries,
            )
            content = self._llm_call(member, system_msg, user_msg)
            duration = time.monotonic() - start

            logger.info(
                "Council member %s (%s) completed in %.1fs",
                role.value, member.model_name, duration,
            )
            return CouncilOutput(
                role=role,
                model_name=member.model_name,
                content=content,
                duration_seconds=duration,
            )

        except RateLimitTimeout as e:
            duration = time.monotonic() - start
            logger.warning("Rate limit timeout for %s: %s", role.value, e)
            return CouncilOutput(
                role=role,
                model_name=member.model_name,
                duration_seconds=duration,
                failed=True,
                error=f"Rate limit timeout: {e}",
            )
        except Exception as e:
            duration = time.monotonic() - start
            logger.exception("Council member %s failed", role.value)
            return CouncilOutput(
                role=role,
                model_name=member.model_name,
                duration_seconds=duration,
                failed=True,
                error=str(e),
            )

    def _build_prompts(
        self,
        role: CouncilRole,
        source_content: str,
        source_metadata: Dict[str, str],
        existing_entities: Optional[List[str]] = None,
        entity_list: str = "",
        existing_page_summaries: str = "",
    ) -> tuple:
        """Build system + user prompt pair for a council member role."""
        if role == CouncilRole.ENTITY_EXTRACTOR:
            context = ""
            if existing_entities:
                context = ENTITY_EXTRACTOR_EXISTING_CONTEXT.format(
                    entity_list="\n".join(f"- {e}" for e in existing_entities[:100])
                )
            return (
                ENTITY_EXTRACTOR_SYSTEM,
                ENTITY_EXTRACTOR_USER.format(
                    source_text=source_content[:12000],
                    existing_entities_context=context,
                ),
            )

        if role == CouncilRole.PROSE_SYNTHESIZER:
            return (
                PROSE_SYNTHESIZER_SYSTEM,
                PROSE_SYNTHESIZER_USER.format(
                    source_text=source_content[:10000],
                    source_title=source_metadata.get("title", "Unknown"),
                    source_type=source_metadata.get("source_type", "article"),
                    source_ref=source_metadata.get("source_ref", ""),
                ),
            )

        if role == CouncilRole.CROSS_CONNECTOR:
            return (
                CROSS_CONNECTOR_SYSTEM,
                CROSS_CONNECTOR_USER.format(
                    source_text=source_content[:8000],
                    entity_list=entity_list or "(no entities extracted)",
                ),
            )

        if role == CouncilRole.CONTRADICTION_FINDER:
            return (
                CONTRADICTION_FINDER_SYSTEM,
                CONTRADICTION_FINDER_USER.format(
                    source_text=source_content[:8000],
                    existing_page_summaries=existing_page_summaries[:8000]
                    or "(no existing pages to compare against)",
                ),
            )

        raise ValueError(f"No prompt template for role: {role}")

    def _llm_call(self, member: CouncilMember, system_msg: str, user_msg: str) -> str:
        """Make an LLM call through the unified client, respecting rate limits."""
        from helper_functions.llm_client import get_client

        # Estimate tokens (rough: 1 token ~= 4 chars)
        estimated_tokens = (len(system_msg) + len(user_msg)) // 4 + member.max_tokens
        self.rate_limiter.acquire(member.model_name, estimated_tokens)

        client = get_client(provider=self.provider, model_name=member.model_name)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        response = client.chat_completion(
            messages=messages,
            temperature=member.temperature,
            max_tokens=member.max_tokens,
        )
        return response or ""

    def cache_session(self, session: CouncilSession, cache_dir: Path) -> Path:
        """Cache a council session for recovery after chairman failure."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        timestamp = session.created_at.strftime("%Y%m%d_%H%M%S")
        cache_path = cache_dir / f"session_{timestamp}.json"

        data = {
            "source_metadata": session.source_metadata,
            "council_mode": session.council_mode,
            "created_at": session.created_at.isoformat(),
            "outputs": [
                {
                    "role": o.role.value,
                    "model_name": o.model_name,
                    "content": o.content,
                    "failed": o.failed,
                    "error": o.error,
                    "duration_seconds": o.duration_seconds,
                }
                for o in session.outputs
            ],
        }
        cache_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Cached council session to %s", cache_path)
        return cache_path


def _get_output_text(session: CouncilSession, role: CouncilRole) -> str:
    """Get the text output from a council member, or a placeholder if failed."""
    output = session.get_output(role)
    if output is None:
        return "(this specialist was not run in this council mode)"
    if output.failed:
        return f"(this specialist failed: {output.error})"
    return output.content


def _safe_parse_json(text: str) -> Optional[Dict]:
    """Parse JSON from LLM output, handling markdown fences and extra text."""
    # Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Try to find a JSON object
    start = cleaned.find("{")
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None
