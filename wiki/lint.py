"""
Wiki Linter - periodic health checks for the wiki vault.

Checks: frontmatter completeness, wikilink validity, orphan pages,
source count consistency, confidence upgrades, stale tensions,
entity registry consistency.

Run weekly alongside newsletter cron, or on-demand via /lint.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import yaml

from wiki.entities import EntityRegistry
from wiki.page_writer import PageWriter, _atomic_write

logger = logging.getLogger(__name__)

REQUIRED_FRONTMATTER = {"title", "type", "created", "updated", "sources", "tags"}


@dataclass
class LintIssue:
    """A single lint finding."""

    page: str
    severity: Literal["high", "medium", "low"]
    category: str
    message: str
    auto_fixable: bool = False
    fixed: bool = False


@dataclass
class LintReport:
    """Complete lint report."""

    issues: List[LintIssue] = field(default_factory=list)
    pages_checked: int = 0
    auto_fixed: int = 0
    needs_review: List[str] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "high")

    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "medium")


class WikiLinter:
    """
    Full wiki health checker.

    7 check categories:
    1. Frontmatter completeness
    2. Wikilink validity
    3. Orphan detection (0 inbound links)
    4. Source count consistency
    5. Confidence upgrades (low->medium when sources >= 2)
    6. Stale tensions (>30 days unresolved) [placeholder for future]
    7. Entity registry consistency
    """

    def __init__(
        self,
        vault_root: Path,
        registry: EntityRegistry,
        page_writer: Optional[PageWriter] = None,
    ) -> None:
        self.vault_root = vault_root
        self.registry = registry
        self.page_writer = page_writer or PageWriter(vault_root)

    def run(self, auto_fix: bool = True) -> LintReport:
        """Run all lint checks. Returns a report."""
        report = LintReport()

        pages = self._scan_all_pages()
        report.pages_checked = len(pages)

        for page_path in pages:
            report.issues.extend(self._check_frontmatter(page_path))
            report.issues.extend(self._check_source_count(page_path))
            report.issues.extend(self._check_confidence_upgrade(page_path))

        report.issues.extend(self._check_wikilinks(pages))
        report.issues.extend(self._detect_orphans(pages))
        report.issues.extend(self._check_registry_consistency(pages))

        if auto_fix:
            for issue in report.issues:
                if issue.auto_fixable and not issue.fixed:
                    self._auto_fix(issue)
                    if issue.fixed:
                        report.auto_fixed += 1

        report.needs_review = [
            i.page for i in report.issues
            if not i.auto_fixable and i.severity in ("high", "medium")
        ]

        logger.info(
            "Lint complete: %d pages, %d issues (%d auto-fixed, %d need review)",
            report.pages_checked,
            report.issue_count,
            report.auto_fixed,
            len(report.needs_review),
        )
        return report

    def _check_frontmatter(self, page_path: Path) -> List[LintIssue]:
        """Check that all required frontmatter fields are present."""
        issues = []
        metadata = self.page_writer.read_page_metadata(page_path)

        if metadata is None:
            issues.append(LintIssue(
                page=page_path.stem,
                severity="high",
                category="frontmatter",
                message="Missing or unparseable frontmatter",
            ))
            return issues

        # Check required fields
        text = page_path.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        if fm_match:
            try:
                data = yaml.safe_load(fm_match.group(1)) or {}
                missing = REQUIRED_FRONTMATTER - set(data.keys())
                for field_name in missing:
                    issues.append(LintIssue(
                        page=page_path.stem,
                        severity="medium",
                        category="frontmatter",
                        message=f"Missing frontmatter field: {field_name}",
                        auto_fixable=field_name in ("last_updated",),
                    ))
            except yaml.YAMLError:
                pass

        return issues

    def _check_wikilinks(self, pages: List[Path]) -> List[LintIssue]:
        """Check that all wikilinks resolve to existing pages."""
        issues = []
        wikilink_pattern = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")

        # Build set of known page slugs
        known_slugs = set()
        for page in pages:
            known_slugs.add(page.stem)
            try:
                known_slugs.add(str(page.relative_to(self.vault_root / "wiki").with_suffix("")))
            except ValueError:
                pass
            # Also add display-name-style references
            known_slugs.add(page.stem.replace("-", " ").lower())

        for page_path in pages:
            content = page_path.read_text(encoding="utf-8")
            for match in wikilink_pattern.finditer(content):
                target = match.group(1).strip()
                target_slug = target.lower().replace(" ", "-")
                target_lower = target.lower()

                if target_slug not in known_slugs and target_lower not in known_slugs and target not in known_slugs:
                    issues.append(LintIssue(
                        page=page_path.stem,
                        severity="low",
                        category="wikilink",
                        message=f"Broken wikilink: [[{target}]]",
                    ))

        return issues

    def _detect_orphans(self, pages: List[Path]) -> List[LintIssue]:
        """Find pages with zero inbound wikilinks."""
        issues = []
        wikilink_pattern = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")

        # Count inbound links per page
        inbound_counts: Dict[str, int] = {p.stem: 0 for p in pages}

        for page_path in pages:
            content = page_path.read_text(encoding="utf-8")
            for match in wikilink_pattern.finditer(content):
                target = match.group(1).strip().lower().replace(" ", "-")
                if "/" in target:
                    target = Path(target).stem
                if target in inbound_counts:
                    inbound_counts[target] += 1

        for slug, count in inbound_counts.items():
            if count == 0:
                issues.append(LintIssue(
                    page=slug,
                    severity="low",
                    category="orphan",
                    message="Orphan page: no inbound wikilinks",
                ))

        return issues

    def _check_source_count(self, page_path: Path) -> List[LintIssue]:
        """Verify sources count in frontmatter matches actual source list."""
        issues = []
        metadata = self.page_writer.read_page_metadata(page_path)
        if metadata is None:
            return issues

        body = self.page_writer.read_page_body(page_path)
        # Count source entries in ## Sources section
        sources_match = re.search(r"## Sources\n(.*?)(?:\n##|\Z)", body, re.DOTALL)
        if sources_match:
            source_lines = [
                l for l in sources_match.group(1).strip().split("\n")
                if l.strip().startswith("-")
            ]
            actual_count = len(source_lines)
            if actual_count != metadata.sources and actual_count > 0:
                issues.append(LintIssue(
                    page=page_path.stem,
                    severity="low",
                    category="source_count",
                    message=f"Source count mismatch: frontmatter says {metadata.sources}, "
                            f"actual list has {actual_count}",
                    auto_fixable=True,
                ))

        return issues

    def _check_confidence_upgrade(self, page_path: Path) -> List[LintIssue]:
        """Check if low-confidence pages can be upgraded."""
        issues = []
        metadata = self.page_writer.read_page_metadata(page_path)
        if metadata is None:
            return issues

        text = page_path.read_text(encoding="utf-8")
        fm_match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        has_confidence = bool(fm_match and re.search(r"^confidence:", fm_match.group(1), re.MULTILINE))
        if not has_confidence:
            return issues

        if metadata.confidence == "low" and metadata.sources >= 2:
            issues.append(LintIssue(
                page=page_path.stem,
                severity="low",
                category="confidence",
                message=f"Eligible for upgrade: {metadata.sources} sources, still confidence=low",
                auto_fixable=True,
            ))

        return issues

    def _check_registry_consistency(self, pages: List[Path]) -> List[LintIssue]:
        """Check that all pages have matching registry entries."""
        issues = []
        registered_slugs = set(self.registry.get_all_slugs())

        for page_path in pages:
            if page_path.stem not in registered_slugs:
                issues.append(LintIssue(
                    page=page_path.stem,
                    severity="medium",
                    category="registry",
                    message="Page exists but has no entity registry entry",
                ))

        return issues

    def _auto_fix(self, issue: LintIssue) -> None:
        """Attempt to auto-fix a lint issue."""
        if issue.category == "source_count":
            self._fix_source_count(issue)
        elif issue.category == "confidence":
            self._fix_confidence_upgrade(issue)
        elif issue.category == "frontmatter" and "updated" in issue.message:
            self._fix_updated(issue)

    def _fix_source_count(self, issue: LintIssue) -> None:
        """Fix source count mismatch in frontmatter."""
        page_path = self._find_page(issue.page)
        if not page_path:
            return

        text = page_path.read_text(encoding="utf-8")
        body = self.page_writer.read_page_body(page_path)

        sources_match = re.search(r"## Sources\n(.*?)(?:\n##|\Z)", body, re.DOTALL)
        if sources_match:
            source_lines = [l for l in sources_match.group(1).strip().split("\n") if l.strip().startswith("-")]
            actual_count = len(source_lines)

            # Update frontmatter
            text = re.sub(
                r"(sources:\s*)\d+",
                f"\\g<1>{actual_count}",
                text,
                count=1,
            )
            _atomic_write(page_path, text)
            issue.fixed = True

    def _fix_confidence_upgrade(self, issue: LintIssue) -> None:
        """Upgrade confidence from low to medium."""
        page_path = self._find_page(issue.page)
        if not page_path:
            return

        text = page_path.read_text(encoding="utf-8")
        text = re.sub(
            r"(confidence:\s*)low",
            "\\g<1>medium",
            text,
            count=1,
        )
        _atomic_write(page_path, text)
        issue.fixed = True

    def _fix_updated(self, issue: LintIssue) -> None:
        """Add missing updated to frontmatter."""
        from datetime import date
        page_path = self._find_page(issue.page)
        if not page_path:
            return

        text = page_path.read_text(encoding="utf-8")
        # Insert last_updated before the closing ---
        text = re.sub(
            r"\n---\n",
            f"\nupdated: {date.today().isoformat()}\n---\n",
            text,
            count=1,
        )
        _atomic_write(page_path, text)
        issue.fixed = True

    def _find_page(self, slug: str) -> Optional[Path]:
        """Find a page path by slug across all wiki directories."""
        for subdir in ["concepts", "entities", "summaries"]:
            page_path = self.vault_root / "wiki" / subdir / f"{slug}.md"
            if page_path.exists():
                return page_path
        return None

    def _scan_all_pages(self) -> List[Path]:
        """Find all wiki pages."""
        pages = []
        for subdir in ["concepts", "entities", "summaries"]:
            dir_path = self.vault_root / "wiki" / subdir
            if dir_path.exists():
                pages.extend(dir_path.rglob("*.md"))
        return pages


def format_lint_telegram(report: LintReport) -> str:
    """Format lint report for Telegram message."""
    lines = [
        "Wiki Lint Report",
        f"  Pages checked: {report.pages_checked}",
        f"  Issues found: {report.issue_count}",
        f"  Auto-fixed: {report.auto_fixed}",
        f"  Needs review: {len(report.needs_review)}",
    ]

    if report.high_count > 0:
        lines.append(f"  HIGH severity: {report.high_count}")

    if report.needs_review:
        lines.append("")
        lines.append("Pages needing review:")
        for page in report.needs_review[:10]:
            lines.append(f"  - {page}")

    return "\n".join(lines)
