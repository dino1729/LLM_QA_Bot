"""
Page Writer for the LLM Wiki.

Assembles and updates individual wiki pages. Handles YAML frontmatter
rendering, section manipulation, and XML diff application from the
chairman's update instructions. All file writes are atomic.
"""
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml

logger = logging.getLogger(__name__)

ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class PageMetadata:
    """YAML frontmatter for a wiki page."""

    title: str
    entity_type: Literal["concept", "person", "source", "analysis"]
    category: str
    tags: tuple = ()
    sources: int = 0
    people: tuple = ()
    last_updated: str = ""
    confidence: ConfidenceLevel = "low"
    wiki_version: int = 1


@dataclass
class PageSections:
    """Structured content sections of a wiki page."""

    core_idea: str = ""
    key_insights: List[str] = field(default_factory=list)
    connections_related: List[str] = field(default_factory=list)
    connections_tensions: List[str] = field(default_factory=list)
    connections_applied: List[str] = field(default_factory=list)
    sources_list: List[str] = field(default_factory=list)
    # Person-specific
    overview: str = ""
    key_ideas: List[str] = field(default_factory=list)
    notable_quotes: List[str] = field(default_factory=list)
    # Analysis-specific
    question: str = ""
    synthesis: str = ""
    sources_cited: List[str] = field(default_factory=list)
    filed_from: str = ""


class PageWriter:
    """Assembles and updates individual wiki pages."""

    def __init__(self, vault_root: Path) -> None:
        self.vault_root = vault_root

    def create_page(
        self,
        page_path: Path,
        metadata: PageMetadata,
        content_markdown: str,
    ) -> None:
        """
        Create a new wiki page from chairman output.

        The content_markdown is the full page body (everything after frontmatter)
        as produced by the chairman. We prepend the rendered frontmatter.
        """
        frontmatter = self.render_frontmatter(metadata)
        full_content = f"{frontmatter}\n{content_markdown.strip()}\n"

        page_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(page_path, full_content)
        logger.info("Created wiki page: %s", page_path.name)

    def create_page_from_full_markdown(
        self,
        page_path: Path,
        full_markdown: str,
    ) -> None:
        """
        Write a complete page (frontmatter + body) as-is from chairman output.
        Used during migration where the chairman produces the complete page.
        """
        page_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(page_path, full_markdown.strip() + "\n")
        logger.info("Created wiki page (full): %s", page_path.name)

    def update_page(
        self,
        page_path: Path,
        xml_diff: str,
    ) -> bool:
        """
        Apply an XML diff from the chairman to an existing page.

        The chairman produces structured XML describing what to add/change.
        This method parses that XML and applies changes surgically,
        preserving existing content.

        Returns True if changes were applied, False if parsing failed.
        """
        if not page_path.exists():
            logger.warning("Cannot update non-existent page: %s", page_path)
            return False

        existing = page_path.read_text(encoding="utf-8")

        try:
            updated = self._apply_xml_diff(existing, xml_diff)
        except (ET.ParseError, ValueError):
            logger.exception("Failed to parse XML diff for %s", page_path.name)
            return False

        if updated != existing:
            _atomic_write(page_path, updated)
            logger.info("Updated wiki page: %s", page_path.name)
            return True

        return False

    def _apply_xml_diff(self, existing_md: str, xml_diff: str) -> str:
        """
        Parse the chairman's XML diff and apply it to the existing markdown.

        Handles: frontmatter updates, section item additions, subsection additions.
        """
        # Extract the <wiki_update> block from potentially mixed content
        match = re.search(r"<wiki_update>(.*?)</wiki_update>", xml_diff, re.DOTALL)
        if not match:
            logger.warning("No <wiki_update> block found in chairman output")
            return existing_md

        xml_content = f"<wiki_update>{match.group(1)}</wiki_update>"
        root = ET.fromstring(xml_content)

        result = existing_md

        # Apply frontmatter updates
        fm_elem = root.find("update_frontmatter")
        if fm_elem is not None:
            result = self._update_frontmatter(result, fm_elem)

        # Apply section updates
        for section_elem in root.findall("section"):
            section_name = section_elem.get("name", "")
            result = self._update_section(result, section_name, section_elem)

        return result

    def _update_frontmatter(self, md: str, fm_elem: ET.Element) -> str:
        """Update YAML frontmatter fields based on XML instructions."""
        fm_match = re.match(r"^---\n(.*?)\n---", md, re.DOTALL)
        if not fm_match:
            return md

        fm_text = fm_match.group(1)
        try:
            fm_data = yaml.safe_load(fm_text) or {}
        except yaml.YAMLError:
            return md

        # Process each frontmatter instruction
        for child in fm_elem:
            tag = child.tag
            text = (child.text or "").strip()

            if tag == "sources" and "INCREMENT" in text:
                fm_data["sources"] = fm_data.get("sources", 0) + 1
            elif tag == "last_updated":
                fm_data["last_updated"] = text or date.today().isoformat()
            elif tag == "people" and text.startswith("ADD"):
                new_people = [p.strip() for p in text.replace("ADD", "").split(",") if p.strip()]
                existing = fm_data.get("people", []) or []
                fm_data["people"] = list(dict.fromkeys(existing + new_people))
            elif tag == "tags" and text.startswith("ADD"):
                new_tags = [t.strip() for t in text.replace("ADD", "").split(",") if t.strip()]
                existing = fm_data.get("tags", []) or []
                fm_data["tags"] = list(dict.fromkeys(existing + new_tags))
            elif tag == "confidence":
                fm_data["confidence"] = text
            elif tag == "wiki_version":
                fm_data["wiki_version"] = fm_data.get("wiki_version", 1) + 1

        # Rebuild frontmatter
        new_fm = yaml.dump(fm_data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return f"---\n{new_fm.strip()}\n---{md[fm_match.end():]}"

    def _update_section(self, md: str, section_name: str, section_elem: ET.Element) -> str:
        """Add items to a named section in the markdown."""
        # Find the section header
        section_pattern = rf"(## {re.escape(section_name)}\n)"
        section_match = re.search(section_pattern, md)
        if not section_match:
            # Section doesn't exist - append it before ## Sources
            sources_match = re.search(r"\n## Sources\n", md)
            if sources_match:
                new_section = f"\n## {section_name}\n"
                md = md[:sources_match.start()] + new_section + md[sources_match.start():]
                section_match = re.search(section_pattern, md)
            else:
                # Append at end
                md += f"\n## {section_name}\n"
                section_match = re.search(section_pattern, md)

        if not section_match:
            return md

        # Find the end of this section (next ## or end of file)
        section_start = section_match.end()
        next_section = re.search(r"\n## ", md[section_start:])
        section_end = section_start + next_section.start() if next_section else len(md)

        section_content = md[section_start:section_end]

        # Process subsections
        for subsection_elem in section_elem.findall("subsection"):
            sub_name = subsection_elem.get("name", "")
            sub_pattern = rf"(### {re.escape(sub_name)}\n)"
            sub_match = re.search(sub_pattern, section_content)

            if sub_match:
                # Find end of subsection
                sub_start = sub_match.end()
                next_sub = re.search(r"\n### ", section_content[sub_start:])
                sub_end = sub_start + next_sub.start() if next_sub else len(section_content)

                # Add items before end of subsection
                new_items = ""
                for item in subsection_elem.findall("add_item"):
                    item_text = (item.text or "").strip()
                    if item_text and item_text not in section_content:
                        new_items += f"- {item_text}\n"

                if new_items:
                    section_content = (
                        section_content[:sub_end].rstrip()
                        + "\n"
                        + new_items
                        + section_content[sub_end:]
                    )

        # Process direct add_item children (not in subsections)
        new_items = ""
        for item in section_elem.findall("add_item"):
            item_text = (item.text or "").strip()
            if item_text and item_text not in section_content:
                new_items += f"- {item_text}\n"

        if new_items:
            section_content = section_content.rstrip() + "\n" + new_items

        return md[:section_start] + section_content + md[section_end:]

    def render_frontmatter(self, metadata: PageMetadata) -> str:
        """Render YAML frontmatter from PageMetadata."""
        fm = {
            "title": metadata.title,
            "type": metadata.entity_type,
            "category": metadata.category,
            "tags": list(metadata.tags),
            "sources": metadata.sources,
            "people": list(metadata.people),
            "last_updated": metadata.last_updated or date.today().isoformat(),
            "confidence": metadata.confidence,
            "wiki_version": metadata.wiki_version,
        }
        rendered = yaml.dump(fm, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return f"---\n{rendered.strip()}\n---\n"

    def read_page_metadata(self, page_path: Path) -> Optional[PageMetadata]:
        """Read and parse frontmatter from an existing page."""
        if not page_path.exists():
            return None

        text = page_path.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if not fm_match:
            return None

        try:
            data = yaml.safe_load(fm_match.group(1)) or {}
            raw_sources = data.get("sources", 0)
            if isinstance(raw_sources, list):
                source_count = len(raw_sources)
            elif isinstance(raw_sources, int):
                source_count = raw_sources
            elif raw_sources:
                source_count = 1
            else:
                source_count = 0

            raw_type = data.get("type", "concept")
            entity_type = {
                "entity": "person",
                "summary": "source",
            }.get(raw_type, raw_type)

            return PageMetadata(
                title=data.get("title", page_path.stem),
                entity_type=entity_type,
                category=data.get("category", ""),
                tags=tuple(data.get("tags", [])),
                sources=source_count,
                people=tuple(data.get("people", [])),
                last_updated=data.get("last_updated", data.get("updated", "")),
                confidence=data.get("confidence", "low"),
                wiki_version=data.get("wiki_version", 1),
            )
        except yaml.YAMLError:
            logger.exception("Failed to parse frontmatter: %s", page_path)
            return None

    def read_page_body(self, page_path: Path) -> str:
        """Read page content without frontmatter."""
        if not page_path.exists():
            return ""

        text = page_path.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n.*?\n---\n?", text, re.DOTALL)
        if fm_match:
            return text[fm_match.end():]
        return text

    def get_one_line_summary(self, page_path: Path) -> str:
        """Extract Core Idea section as a one-line summary for index.md."""
        body = self.read_page_body(page_path)
        # Find ## Core Idea section
        match = re.search(r"## Core Idea\n(.*?)(?:\n##|\Z)", body, re.DOTALL)
        if match:
            summary = match.group(1).strip()
            # Take first sentence
            first_sentence = summary.split(". ")[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."
            return first_sentence[:200]
        return page_path.stem.replace("-", " ").title()


def _atomic_write(path: Path, content: str) -> None:
    """Write content to file atomically via tmp + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
