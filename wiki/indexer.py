"""
Wiki Indexer - maintains the llm-wiki skill index and operation logs.

Manages: wiki/index.md and log/YYYYMMDD.md. Compatibility helpers for
connections/timeline/overview write inside the skill layout instead of
recreating the older vault-level files.
"""
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from wiki.page_writer import PageWriter, _atomic_write

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexEntry:
    """A single entry in index.md."""

    title: str
    relative_path: str
    one_line_summary: str
    entity_type: str
    source_count: int


@dataclass(frozen=True)
class LogEntry:
    """A single entry in log.md."""

    timestamp: str
    operation: str
    pages_affected: str
    details: str


class WikiIndexer:
    """
    Maintains the llm-wiki skill navigation and log files.

    - wiki/index.md: master catalog, updated on every ingest/compile
    - log/YYYYMMDD.md: append-only daily operation log
    - wiki/concepts/knowledge-connections.md: named concept clusters
    - outputs/queries/wiki-growth-timeline.md: growth snapshots
    - wiki/concepts/knowledge-overview.md: high-level synthesis
    """

    def __init__(self, vault_root: Path, page_writer: Optional[PageWriter] = None) -> None:
        self.vault_root = vault_root
        self.page_writer = page_writer or PageWriter(vault_root)
        self._log_lock = threading.Lock()

    @property
    def index_path(self) -> Path:
        return self.vault_root / "wiki" / "index.md"

    @property
    def log_dir(self) -> Path:
        return self.vault_root / "log"

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"{datetime.now().strftime('%Y%m%d')}.md"

    @property
    def connections_path(self) -> Path:
        return self.vault_root / "wiki" / "concepts" / "knowledge-connections.md"

    @property
    def timeline_path(self) -> Path:
        return self.vault_root / "outputs" / "queries" / "wiki-growth-timeline.md"

    @property
    def overview_path(self) -> Path:
        return self.vault_root / "wiki" / "concepts" / "knowledge-overview.md"

    def update_index(
        self,
        updated_pages: Optional[List[Path]] = None,
        created_pages: Optional[List[Path]] = None,
    ) -> None:
        """
        Update index.md with entries for changed/new pages.

        For efficiency, only rewrites entries for specified pages.
        If no pages specified, rebuilds the entire index.
        """
        # The installed llm-wiki skill requires every generated page to appear
        # exactly once. Rebuild from disk each time so incremental updates never
        # drop pre-existing pages when the index format changes.
        existing = {}
        pages_to_index = self._scan_all_pages()

        for page_path in pages_to_index:
            if not page_path.exists():
                # Page was deleted - remove from index
                rel_path = self._relative_path(page_path)
                existing.pop(rel_path, None)
                continue

            metadata = self.page_writer.read_page_metadata(page_path)
            if metadata is None:
                continue

            summary = self.page_writer.get_one_line_summary(page_path)
            rel_path = self._relative_path(page_path)
            existing[rel_path] = IndexEntry(
                title=metadata.title,
                relative_path=rel_path,
                one_line_summary=summary,
                entity_type=metadata.entity_type,
                source_count=metadata.sources,
            )

        self._write_index(existing)
        logger.info("Updated index.md: %d entries", len(existing))

    def append_log(self, entry: LogEntry) -> None:
        """Append a single entry to log/YYYYMMDD.md. Thread-safe via lock."""
        try:
            parsed = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now()

        log_path = self.log_dir / f"{parsed.strftime('%Y%m%d')}.md"
        header = f"# {parsed.strftime('%Y-%m-%d')}\n\n"
        line = f"## [{parsed.strftime('%H:%M')}] {entry.operation} | {entry.pages_affected}\n"
        if entry.details:
            line += f"- {entry.details}\n"

        with self._log_lock:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if not log_path.exists():
                log_path.write_text(header, encoding="utf-8")

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n" + line)

    def make_log_entry(
        self,
        operation: str,
        pages: List[str],
        details: str = "",
    ) -> LogEntry:
        """Create a log entry with current timestamp."""
        return LogEntry(
            timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            operation=operation,
            pages_affected=", ".join(pages[:5]) + (f" (+{len(pages) - 5} more)" if len(pages) > 5 else ""),
            details=details,
        )

    def update_connections(self, new_links: Optional[List[tuple]] = None) -> None:
        """
        Update wiki/concepts/knowledge-connections.md with concept clusters.

        Scans all wiki pages for wikilinks, builds a graph,
        and identifies clusters of densely connected concepts.
        """
        # Build link graph from all wiki pages
        link_graph = self._build_link_graph()

        # Identify clusters (connected components with 3+ nodes)
        clusters = self._find_clusters(link_graph)

        # Write CONNECTIONS.md
        lines = [
            "# Knowledge Connections\n",
            "\nConcept clusters identified by wikilink density.\n",
            "Auto-updated by the wiki pipeline.\n\n---\n",
        ]

        for i, cluster in enumerate(clusters, 1):
            if len(cluster["members"]) < 3:
                continue
            lines.append(f"\n## {cluster['name']}\n")
            if cluster.get("theme"):
                lines.append(f"**Core theme**: {cluster['theme']}\n")
            for member in sorted(cluster["members"]):
                lines.append(f"- [[{member}]]\n")

        if not any(len(c["members"]) >= 3 for c in clusters):
            lines.append("\n*No clusters with 3+ members yet. The wiki will develop clusters as more content is added.*\n")

        self.connections_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.connections_path, "".join(lines))
        logger.info("Updated knowledge-connections.md: %d clusters", len(clusters))

    def update_timeline(self) -> None:
        """
        Append a growth snapshot to outputs/queries/wiki-growth-timeline.md.
        Called weekly alongside lint.
        """
        stats = self._compute_stats()
        now = datetime.now().strftime("%Y-%m-%d")

        entry = (
            f"\n## {now}\n"
            f"- Pages: {stats['total_pages']} "
            f"(concepts: {stats['concepts']}, entities: {stats['entities']}, "
            f"summaries: {stats['summaries']})\n"
            f"- Connections: {stats['total_links']} wikilinks\n"
            f"- Avg sources per concept: {stats['avg_sources']:.1f}\n"
        )

        if not self.timeline_path.exists():
            self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
            self.timeline_path.write_text(
                "# Wiki Growth Timeline\n\n"
                "Weekly snapshots of wiki evolution.\n\n---\n",
                encoding="utf-8",
            )

        with open(self.timeline_path, "a", encoding="utf-8") as f:
            f.write(entry)

        logger.info("Updated wiki growth timeline: %d total pages", stats["total_pages"])

    def _load_index_entries(self) -> Dict[str, IndexEntry]:
        """Parse existing index.md into a dict keyed by relative path."""
        entries = {}
        if not self.index_path.exists():
            return entries

        content = self.index_path.read_text(encoding="utf-8")
        # Parse lines like: - [Title](path) - summary (type, N sources)
        pattern = re.compile(
            r"^- \[(.+?)\]\((.+?)\) - (.+?) \((\w+), (\d+) sources?\)$",
            re.MULTILINE,
        )
        for match in pattern.finditer(content):
            title, rel_path, summary, entity_type, source_count = match.groups()
            entries[rel_path] = IndexEntry(
                title=title,
                relative_path=rel_path,
                one_line_summary=summary,
                entity_type=entity_type,
                source_count=int(source_count),
            )
        return entries

    def _write_index(self, entries: Dict[str, IndexEntry]) -> None:
        """Write wiki/index.md in the installed llm-wiki skill format."""
        lines = [
            "# Index — Memory Palace\n",
            "\n> Personal Memory Palace knowledge base compiled into a persistent LLM-maintained wiki.\n",
            "\n## 🔖 Navigation\n",
            "- [[#Concepts]] · [[#Entities]] · [[#Summaries]] · [[#Open Questions]]\n",
        ]

        # Group by type
        by_type: Dict[str, List[IndexEntry]] = {}
        for entry in entries.values():
            kind = {
                "person": "entity",
                "entity": "entity",
                "source": "summary",
                "summary": "summary",
            }.get(entry.entity_type, "concept")
            by_type.setdefault(kind, []).append(entry)

        lines.append("\n## Concepts\n\n")
        concepts = by_type.get("concept", [])
        if concepts:
            lines.append("### General\n")
            for entry in sorted(concepts, key=lambda e: e.title.lower()):
                lines.append(f"- [[{entry.relative_path}|{entry.title}]] — {entry.one_line_summary}\n")
        else:
            lines.append("*(none yet)*\n")

        lines.append("\n## Entities\n\n")
        entities = by_type.get("entity", [])
        if entities:
            for entry in sorted(entities, key=lambda e: e.title.lower()):
                lines.append(f"- [[{entry.relative_path}|{entry.title}]] — {entry.one_line_summary}\n")
        else:
            lines.append("*(none yet)*\n")

        lines.append("\n## Summaries (chronological)\n\n")
        summaries = by_type.get("summary", [])
        if summaries:
            for entry in sorted(summaries, key=lambda e: e.title.lower()):
                lines.append(f"- [[{entry.relative_path}|{entry.title}]] — {entry.one_line_summary}\n")
        else:
            lines.append("*(none yet)*\n")

        lines.append(
            "\n## Open Questions\n\n"
            "- Which pages need source-backed expansion next?\n"
            "- Which migrated pages need audit for accuracy and link quality?\n"
        )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.index_path, "".join(lines))

    def _scan_all_pages(self) -> Set[Path]:
        """Find generated wiki pages in the llm-wiki skill layout."""
        pages = set()
        for subdir in ["concepts", "entities", "summaries"]:
            dir_path = self.vault_root / "wiki" / subdir
            if dir_path.exists():
                pages.update(dir_path.rglob("*.md"))
        pages.discard(self.index_path)
        return pages

    def _relative_path(self, page_path: Path) -> str:
        """Get wikilink path relative to wiki/ without the .md suffix."""
        try:
            return str(page_path.relative_to(self.vault_root / "wiki").with_suffix(""))
        except ValueError:
            return str(page_path.with_suffix(""))

    def _build_link_graph(self) -> Dict[str, Set[str]]:
        """Build adjacency list of wikilinks between pages."""
        graph: Dict[str, Set[str]] = {}
        wikilink_pattern = re.compile(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]")

        for page_path in self._scan_all_pages():
            page_name = page_path.stem
            content = page_path.read_text(encoding="utf-8")
            links = set()
            for match in wikilink_pattern.finditer(content):
                target = match.group(1).strip().lower().replace(" ", "-")
                if target != page_name:
                    links.add(target)
            graph[page_name] = links

        return graph

    def _find_clusters(self, graph: Dict[str, Set[str]]) -> List[Dict]:
        """Find connected components in the link graph."""
        visited: Set[str] = set()
        clusters = []

        for node in graph:
            if node in visited:
                continue
            # BFS to find connected component
            component: Set[str] = set()
            queue = [node]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in graph.get(current, set()):
                    if neighbor not in visited and neighbor in graph:
                        queue.append(neighbor)

            if len(component) >= 2:
                # Name the cluster after the most-connected node
                hub = max(component, key=lambda n: len(graph.get(n, set()) & component))
                clusters.append({
                    "name": hub.replace("-", " ").title() + " Cluster",
                    "members": sorted(component),
                    "theme": "",
                })

        # Sort by cluster size descending
        clusters.sort(key=lambda c: len(c["members"]), reverse=True)
        return clusters

    def _compute_stats(self) -> Dict:
        """Compute basic wiki statistics."""
        stats = {"concepts": 0, "entities": 0, "summaries": 0, "total_links": 0}
        total_sources = 0
        concept_count = 0

        for subdir, key in [
            ("concepts", "concepts"),
            ("entities", "entities"),
            ("summaries", "summaries"),
        ]:
            dir_path = self.vault_root / "wiki" / subdir
            if dir_path.exists():
                count = len(list(dir_path.rglob("*.md")))
                stats[key] = count

        graph = self._build_link_graph()
        for links in graph.values():
            stats["total_links"] += len(links)

        # Compute avg sources per concept
        concepts_dir = self.vault_root / "wiki" / "concepts"
        if concepts_dir.exists():
            for page in concepts_dir.rglob("*.md"):
                metadata = self.page_writer.read_page_metadata(page)
                if metadata:
                    total_sources += metadata.sources
                    concept_count += 1

        stats["total_pages"] = sum(stats[k] for k in ["concepts", "entities", "summaries"])
        stats["avg_sources"] = total_sources / concept_count if concept_count > 0 else 0
        return stats
