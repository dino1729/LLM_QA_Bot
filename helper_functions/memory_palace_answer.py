"""
Memory Palace Answer Engine - Synthesizes answers from user wisdom and web knowledge.

This module provides:
- Dual-store retrieval (user wisdom first, then web knowledge)
- Confidence tier calculation based on match quality
- Answer synthesis with source attribution
- Research suggestion when knowledge is partial
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import List, Optional, Tuple

from config import config
from helper_functions.llm_client import get_client
from helper_functions.memory_palace_db import (
    CATEGORIES,
    MemoryPalaceDB,
    SimilarLesson,
)
from helper_functions.web_knowledge_db import (
    ConfidenceTier,
    SimilarWebKnowledge,
    WebKnowledgeDB,
)

logger = logging.getLogger(__name__)


class SourceType(StrEnum):
    """Source of the answer."""
    USER_WISDOM = "user_wisdom"
    WEB_KNOWLEDGE = "web_knowledge"
    BOTH = "both"
    NONE = "none"


@dataclass
class AnswerResult:
    """Result from the answer engine."""
    answer_text: str
    confidence_tier: ConfidenceTier
    source_type: SourceType
    related_topics: List[str] = field(default_factory=list)
    offer_to_research: bool = False
    reasoning_prefix: str = ""
    # Source details for attribution
    wisdom_matches: List[SimilarLesson] = field(default_factory=list)
    knowledge_matches: List[SimilarWebKnowledge] = field(default_factory=list)


# EDITH persona system prompt
EDITH_SYSTEM_PROMPT = """You are EDITH, a personal knowledge assistant for Dinesh.
You help recall lessons from his Memory Palace and research new topics.
Speak in first person ("I remember...", "I found that...").
Be warm but concise. Focus on insights, not fluff.
When synthesizing answers, prioritize Dinesh's personal lessons over generic web knowledge."""


class AnswerEngine:
    """
    Engine for answering questions from Memory Palace.

    Searches user wisdom first, then web knowledge. Synthesizes answers
    with confidence tiers and source attribution.
    """

    # Thresholds for confidence calculation
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    PARTIAL_MATCH_THRESHOLD = 0.4

    def __init__(
        self,
        wisdom_db: MemoryPalaceDB = None,
        knowledge_db: WebKnowledgeDB = None,
        provider: str = None,
        model_tier: str = None,
    ):
        """
        Initialize the answer engine.

        Args:
            wisdom_db: User wisdom database (Memory Palace)
            knowledge_db: Web knowledge database
            provider: LLM provider for synthesis
            model_tier: Model tier for synthesis
        """
        self.wisdom_db = wisdom_db or MemoryPalaceDB()
        self.knowledge_db = knowledge_db or WebKnowledgeDB()
        self.provider = provider or config.memory_palace_provider
        self.model_tier = model_tier or "smart"  # Use smart model for synthesis

    def answer(
        self,
        question: str,
        session_context: List[dict] = None,
        search_web_knowledge: bool = True,
    ) -> AnswerResult:
        """
        Answer a question from stored knowledge.

        Args:
            question: The user's question
            session_context: Previous conversation turns for context
            search_web_knowledge: Whether to also search web knowledge store

        Returns:
            AnswerResult with answer, confidence, and source info
        """
        session_context = session_context or []

        # Step 1: Search user wisdom first (always)
        wisdom_results = self.wisdom_db.find_similar(question, top_k=5)

        # Step 2: Optionally search web knowledge
        knowledge_results = []
        if search_web_knowledge:
            knowledge_results = self.knowledge_db.find_similar(question, top_k=3)

        # Step 3: Determine best match and confidence
        best_wisdom = wisdom_results[0] if wisdom_results else None
        best_knowledge = knowledge_results[0] if knowledge_results else None

        # Calculate confidence based on match quality
        confidence, source_type, offer_research = self._calculate_confidence(
            best_wisdom, best_knowledge
        )

        # Step 4: Generate answer based on what we found
        if source_type == SourceType.NONE:
            # No relevant knowledge found
            related = self._find_related_topics(question)
            return AnswerResult(
                answer_text="",
                confidence_tier=ConfidenceTier.UNCERTAIN,
                source_type=SourceType.NONE,
                related_topics=related,
                offer_to_research=True,
                reasoning_prefix="[No matching knowledge]",
                wisdom_matches=[],
                knowledge_matches=[],
            )

        # Step 5: Synthesize answer
        answer_text, reasoning_prefix = self._synthesize_answer(
            question,
            wisdom_results[:3] if wisdom_results else [],
            knowledge_results[:2] if knowledge_results else [],
            session_context,
            source_type,
            confidence,
        )

        return AnswerResult(
            answer_text=answer_text,
            confidence_tier=confidence,
            source_type=source_type,
            related_topics=[],
            offer_to_research=offer_research,
            reasoning_prefix=reasoning_prefix,
            wisdom_matches=wisdom_results[:3],
            knowledge_matches=knowledge_results[:2],
        )

    def _calculate_confidence(
        self,
        best_wisdom: Optional[SimilarLesson],
        best_knowledge: Optional[SimilarWebKnowledge],
    ) -> Tuple[ConfidenceTier, SourceType, bool]:
        """
        Calculate confidence tier based on match quality.

        Returns:
            (confidence_tier, source_type, offer_to_research)
        """
        wisdom_score = best_wisdom.similarity_score if best_wisdom else 0
        knowledge_score = best_knowledge.similarity_score if best_knowledge else 0

        # Determine primary source
        if wisdom_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceTier.VERY_CONFIDENT, SourceType.USER_WISDOM, False

        if wisdom_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            if knowledge_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                return ConfidenceTier.VERY_CONFIDENT, SourceType.BOTH, False
            return ConfidenceTier.FAIRLY_SURE, SourceType.USER_WISDOM, True

        if knowledge_score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceTier.FAIRLY_SURE, SourceType.WEB_KNOWLEDGE, False

        if knowledge_score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceTier.FAIRLY_SURE, SourceType.WEB_KNOWLEDGE, True

        if wisdom_score >= self.PARTIAL_MATCH_THRESHOLD:
            return ConfidenceTier.UNCERTAIN, SourceType.USER_WISDOM, True

        if knowledge_score >= self.PARTIAL_MATCH_THRESHOLD:
            return ConfidenceTier.UNCERTAIN, SourceType.WEB_KNOWLEDGE, True

        return ConfidenceTier.UNCERTAIN, SourceType.NONE, True

    def _synthesize_answer(
        self,
        question: str,
        wisdom_matches: List[SimilarLesson],
        knowledge_matches: List[SimilarWebKnowledge],
        session_context: List[dict],
        source_type: SourceType,
        confidence: ConfidenceTier,
    ) -> Tuple[str, str]:
        """
        Synthesize an answer using LLM.

        Returns:
            (answer_text, reasoning_prefix)
        """
        client = get_client(provider=self.provider, model_tier=self.model_tier)

        # Build context from matches
        wisdom_context = ""
        if wisdom_matches:
            wisdom_context = "DINESH'S PERSONAL LESSONS (prioritize these):\n"
            for i, match in enumerate(wisdom_matches, 1):
                cat_display = CATEGORIES.get(
                    match.lesson.metadata.category, {}
                ).get("display", match.lesson.metadata.category)
                wisdom_context += (
                    f"{i}. [{cat_display}] {match.lesson.distilled_text} "
                    f"(match: {match.similarity_score:.0%})\n"
                )

        knowledge_context = ""
        if knowledge_matches:
            knowledge_context = "\nWEB KNOWLEDGE (supplementary):\n"
            for i, match in enumerate(knowledge_matches, 1):
                sources = ", ".join(match.knowledge.metadata.source_urls[:2]) or "web"
                knowledge_context += (
                    f"{i}. {match.knowledge.distilled_text} "
                    f"(source: {sources})\n"
                )

        # Build session context
        context_str = ""
        if session_context:
            context_str = "RECENT CONVERSATION:\n"
            for msg in session_context[-6:]:  # Last 3 turns (6 messages)
                role = "Dinesh" if msg.get("role") == "user" else "EDITH"
                context_str += f"{role}: {msg.get('content', '')[:200]}\n"
            context_str += "\n"

        # Confidence guidance
        confidence_guide = {
            ConfidenceTier.VERY_CONFIDENT: "You have strong evidence. Be direct and confident.",
            ConfidenceTier.FAIRLY_SURE: "You have moderate evidence. Be helpful but acknowledge uncertainty.",
            ConfidenceTier.UNCERTAIN: "Evidence is weak. Be honest about limited knowledge.",
        }

        prompt = f"""{EDITH_SYSTEM_PROMPT}

{context_str}{wisdom_context}{knowledge_context}

QUESTION: {question}

GUIDANCE: {confidence_guide.get(confidence, '')}

Answer Dinesh's question based on the knowledge above.
- Lead with the most relevant insight
- Keep it concise (2-4 sentences max)
- If from personal lessons, mention it naturally ("You've noted that..." or "Your lessons suggest...")
- If from web knowledge, attribute briefly ("Research indicates...")

ANSWER:"""

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            answer_text = response.strip()
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            # Fallback to direct quote from best match
            if wisdom_matches:
                answer_text = wisdom_matches[0].lesson.distilled_text
            elif knowledge_matches:
                answer_text = knowledge_matches[0].knowledge.distilled_text
            else:
                answer_text = "I couldn't synthesize an answer at this time."

        # Generate reasoning prefix
        reasoning_prefix = self._generate_reasoning_prefix(
            source_type, confidence, wisdom_matches, knowledge_matches
        )

        return answer_text, reasoning_prefix

    def _generate_reasoning_prefix(
        self,
        source_type: SourceType,
        confidence: ConfidenceTier,
        wisdom_matches: List[SimilarLesson],
        knowledge_matches: List[SimilarWebKnowledge],
    ) -> str:
        """Generate the reasoning prefix shown before answers."""
        confidence_display = {
            ConfidenceTier.VERY_CONFIDENT: "Very confident",
            ConfidenceTier.FAIRLY_SURE: "Fairly sure",
            ConfidenceTier.UNCERTAIN: "Uncertain",
        }

        if source_type == SourceType.USER_WISDOM:
            score = wisdom_matches[0].similarity_score if wisdom_matches else 0
            return f"[From memory: {confidence_display[confidence]}] ({score:.0%} match)"

        if source_type == SourceType.WEB_KNOWLEDGE:
            count = len(knowledge_matches)
            return f"[From research: {confidence_display[confidence]}] ({count} sources)"

        if source_type == SourceType.BOTH:
            return f"[From memory + research: {confidence_display[confidence]}]"

        return "[No matching knowledge]"

    def _find_related_topics(self, question: str, limit: int = 3) -> List[str]:
        """Find related topics when no direct match is found."""
        # Get a broader search with lower threshold
        wisdom_results = self.wisdom_db.find_similar(question, top_k=10)

        related = []
        seen_categories = set()
        for match in wisdom_results:
            if match.similarity_score < 0.2:
                break
            cat = match.lesson.metadata.category
            if cat not in seen_categories:
                cat_display = CATEGORIES.get(cat, {}).get("display", cat)
                related.append(cat_display)
                seen_categories.add(cat)
            if len(related) >= limit:
                break

        return related

    def check_for_conflict(
        self,
        wisdom_lesson: SimilarLesson,
        web_info: str,
    ) -> bool:
        """
        Check if web research contradicts stored wisdom.

        Args:
            wisdom_lesson: The user's stored lesson
            web_info: Newly researched information

        Returns:
            True if there's a potential conflict
        """
        client = get_client(provider=self.provider, model_tier="fast")

        prompt = f"""Compare these two pieces of information and determine if they contradict each other.

USER'S STORED WISDOM: {wisdom_lesson.lesson.distilled_text}

NEW WEB RESEARCH: {web_info}

Do these contradict each other? Answer ONLY "yes" or "no"."""

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            return response.strip().lower().startswith("yes")
        except Exception as e:
            logger.warning(f"Conflict check failed: {e}")
            return False


def format_answer_for_telegram(result: AnswerResult) -> str:
    """
    Format an AnswerResult for Telegram display.

    Args:
        result: The answer result to format

    Returns:
        Formatted string for Telegram message
    """
    if result.source_type == SourceType.NONE:
        msg = "I don't have any knowledge about this topic.\n\n"
        if result.related_topics:
            msg += "Related topics I know about:\n"
            for topic in result.related_topics:
                msg += f"  - {topic}\n"
        return msg

    # Build formatted response
    lines = [result.reasoning_prefix, "", result.answer_text]

    # Add source attribution if from user wisdom
    if result.source_type == SourceType.USER_WISDOM and result.wisdom_matches:
        cat = result.wisdom_matches[0].lesson.metadata.category
        cat_display = CATEGORIES.get(cat, {}).get("display", cat)
        lines.append(f"\n(From your {cat_display} lessons)")

    return "\n".join(lines)
