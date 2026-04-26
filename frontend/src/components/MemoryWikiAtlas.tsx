import { useEffect, useMemo, useState } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import { api, type WikiListResponse, type WikiPageDetail, type WikiPageSummary, type WikiSummaryResponse } from '../api';

interface WikiRoute {
  section: string;
  slug: string;
}

interface Filters {
  section: string;
  category: string;
  tag: string;
  quality: string;
}

const initialFilters: Filters = {
  section: '',
  category: '',
  tag: '',
  quality: '',
};

function parseWikiRoute(pathname = window.location.pathname): WikiRoute | null {
  const parts = pathname.replace(/^\/wiki\/?/, '').split('/').filter(Boolean);
  if (parts.length < 2) return null;
  return {
    section: decodeURIComponent(parts[0]),
    slug: decodeURIComponent(parts[1]),
  };
}

function label(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function qualityLabel(flag: string): string {
  const labels: Record<string, string> = {
    zero_sources: 'No sources',
    low_confidence: 'Low confidence',
    missing_summary: 'No summary',
    missing_category: 'No category',
  };
  return labels[flag] || label(flag);
}

function displayDate(value: string): string {
  if (!value) return 'Unknown';
  return value.slice(0, 10);
}

function sectionName(value: string): string {
  const labels: Record<string, string> = {
    concepts: 'Concepts',
    entities: 'Entities',
    summaries: 'Summaries',
  };
  return labels[value] || label(value);
}

function isAborted(error: unknown): boolean {
  return error instanceof Error && Boolean((error as Error & { aborted?: boolean }).aborted);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function MemoryWikiAtlas() {
  const [route, setRoute] = useState<WikiRoute | null>(() => parseWikiRoute());
  const [summary, setSummary] = useState<WikiSummaryResponse | null>(null);
  const [list, setList] = useState<WikiListResponse | null>(null);
  const [detail, setDetail] = useState<WikiPageDetail | null>(null);
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<Filters>(initialFilters);
  const [error, setError] = useState('');
  const [failedRoute, setFailedRoute] = useState('');

  useEffect(() => {
    const handlePopState = () => setRoute(parseWikiRoute());
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  useEffect(() => {
    const call = api.wiki.summary();
    call.promise
      .then(setSummary)
      .catch((err) => {
        if (isAborted(err)) return;
        setError(errorMessage(err));
      });
    return () => call.cancel();
  }, []);

  useEffect(() => {
    let cancel: (() => void) | null = null;
    const timer = window.setTimeout(() => {
      const call = api.wiki.pages({
        query,
        section: filters.section,
        category: filters.category,
        tag: filters.tag,
        quality: filters.quality,
        limit: 300,
      });
      cancel = call.cancel;
      call.promise
        .then((response) => {
          setList(response);
          setError('');
        })
        .catch((err) => {
          if (isAborted(err)) return;
          setError(errorMessage(err));
        });
    }, 160);

    return () => {
      window.clearTimeout(timer);
      if (cancel) cancel();
    };
  }, [query, filters]);

  useEffect(() => {
    if (!route) {
      return;
    }

    const call = api.wiki.page(route.section, route.slug);
    const routeKey = `${route.section}/${route.slug}`;
    call.promise
      .then((response) => {
        setDetail(response);
        setFailedRoute('');
        setError('');
      })
      .catch((err) => {
        if (isAborted(err)) return;
        setFailedRoute(routeKey);
        setError(errorMessage(err));
      });

    return () => call.cancel();
  }, [route]);

  const pagesById = useMemo(() => {
    const map = new Map<string, WikiPageSummary>();
    list?.pages.forEach((page) => map.set(page.id, page));
    return map;
  }, [list]);

  const groupedPages = useMemo(() => {
    if (!list) return [];
    return list.groups.map((group) => ({
      ...group,
      pages: group.page_ids.map((id) => pagesById.get(id)).filter(Boolean) as WikiPageSummary[],
    }));
  }, [list, pagesById]);

  const navigateTo = (url: string) => {
    window.history.pushState({}, '', url);
    setRoute(parseWikiRoute(url));
  };

  const openPage = (page: WikiPageSummary) => {
    navigateTo(page.url);
  };

  const openAtlasRoot = () => {
    window.history.pushState({}, '', '/wiki');
    setRoute(null);
  };

  const updateFilter = (key: keyof Filters, value: string) => {
    setFilters((current) => ({ ...current, [key]: value }));
  };

  const clearFilters = () => {
    setQuery('');
    setFilters(initialFilters);
  };

  const handleMarkdownLink = (event: React.MouseEvent<HTMLAnchorElement>, href?: string) => {
    if (!href || !href.startsWith('/wiki/')) return;
    event.preventDefault();
    navigateTo(href);
  };

  const hasDetail = Boolean(route);
  const activeDetail = route && detail?.section === route.section && detail.slug === route.slug ? detail : null;
  const routeKey = route ? `${route.section}/${route.slug}` : '';
  const listLoading = !list;
  const detailLoading = Boolean(route && !activeDetail && failedRoute !== routeKey);
  const totalPages = summary?.total_pages ?? 0;

  const markdownComponents: Components = {
    a: ({ href, children, ...props }) => (
      <a href={href} onClick={(event) => handleMarkdownLink(event, href)} {...props}>
        {children}
      </a>
    ),
  };

  return (
    <div className="wiki-atlas">
      <header className="wiki-atlas__masthead">
        <div>
          <p className="wiki-atlas__eyebrow">Memory Wiki Vault</p>
          <h1>Atlas</h1>
          <p className="wiki-atlas__subtitle">
            Browse the generated wiki by concepts, links, sources, and weak spots.
          </p>
        </div>
        <a className="wiki-atlas__hub-link" href="/">Nexus Mind</a>
      </header>

      <section className="wiki-atlas__stats" aria-label="Vault stats">
        <div>
          <span>{totalPages}</span>
          <p>Pages</p>
        </div>
        <div>
          <span>{summary?.total_source_count ?? 0}</span>
          <p>Sources</p>
        </div>
        <div>
          <span>{summary?.total_backlinks ?? 0}</span>
          <p>Backlinks</p>
        </div>
        <div>
          <span>{summary?.weak_page_count ?? 0}</span>
          <p>Badged</p>
        </div>
      </section>

      {error && <div className="wiki-atlas__alert">{error}</div>}

      {summary && summary.total_pages === 0 ? (
        <section className="wiki-atlas__empty">
          <h2>No generated wiki pages found</h2>
          <p>{summary.message}</p>
          <code>{summary.vault_path}</code>
        </section>
      ) : (
        <main className={`wiki-atlas__workspace ${hasDetail ? 'has-detail' : ''}`}>
          <aside className="wiki-atlas__sidebar" aria-label="Wiki pages">
            <div className="wiki-atlas__search">
              <label htmlFor="wiki-search">Search atlas</label>
              <input
                id="wiki-search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search titles, tags, headings..."
                type="text"
              />
            </div>

            <div className="wiki-atlas__filters">
              <select value={filters.section} onChange={(event) => updateFilter('section', event.target.value)}>
                <option value="">All sections</option>
                {(list?.filters.sections || ['concepts', 'entities', 'summaries']).map((section) => (
                  <option key={section} value={section}>{sectionName(section)}</option>
                ))}
              </select>

              <select value={filters.category} onChange={(event) => updateFilter('category', event.target.value)}>
                <option value="">All categories</option>
                {Object.entries(list?.filters.categories || summary?.categories || {}).map(([category, count]) => (
                  <option key={category} value={category}>{category} ({count})</option>
                ))}
              </select>

              <select value={filters.tag} onChange={(event) => updateFilter('tag', event.target.value)}>
                <option value="">All tags</option>
                {Object.entries(list?.filters.tags || summary?.tags || {}).slice(0, 80).map(([tag, count]) => (
                  <option key={tag} value={tag}>{tag} ({count})</option>
                ))}
              </select>

              <select value={filters.quality} onChange={(event) => updateFilter('quality', event.target.value)}>
                <option value="">All quality states</option>
                {Object.entries(list?.filters.quality_flags || summary?.quality_counts || {}).map(([flag, count]) => (
                  <option key={flag} value={flag}>{qualityLabel(flag)} ({count})</option>
                ))}
              </select>
            </div>

            <div className="wiki-atlas__list-head">
              <span>{listLoading ? 'Scanning...' : `${list?.total ?? 0} results`}</span>
              <button type="button" onClick={clearFilters}>Clear</button>
            </div>

            <div className="wiki-atlas__page-list">
              {groupedPages.map((group) => (
                <section key={group.name} className="wiki-atlas__group">
                  <h2>{group.name}</h2>
                  {group.pages.map((page) => (
                    <button
                      className={`wiki-atlas__page-row ${activeDetail?.id === page.id ? 'is-active' : ''}`}
                      key={page.id}
                      onClick={() => openPage(page)}
                      type="button"
                    >
                      <span className="wiki-atlas__row-title">{page.title}</span>
                      <span className="wiki-atlas__row-summary">{page.summary || 'No summary available.'}</span>
                      <span className="wiki-atlas__row-meta">
                        <span>{sectionName(page.section)}</span>
                        <span>{page.source_count} sources</span>
                        <span>{page.backlink_count} backlinks</span>
                      </span>
                      {page.quality_flags.length > 0 && (
                        <span className="wiki-atlas__badges">
                          {page.quality_flags.slice(0, 2).map((flag) => (
                            <span key={flag}>{qualityLabel(flag)}</span>
                          ))}
                        </span>
                      )}
                    </button>
                  ))}
                </section>
              ))}
            </div>
          </aside>

          <article className="wiki-atlas__reader" aria-label="Selected wiki page">
            {hasDetail && (
              <button className="wiki-atlas__back" type="button" onClick={openAtlasRoot}>
                Back to atlas
              </button>
            )}

            {detailLoading && <div className="wiki-atlas__reader-empty">Loading page...</div>}

            {!detailLoading && !activeDetail && (
              <div className="wiki-atlas__reader-empty">
                <h2>Select a page</h2>
                <p>Choose a concept from the atlas to read the page and inspect its source trail.</p>
              </div>
            )}

            {!detailLoading && activeDetail && (
              <>
                <div className="wiki-atlas__reader-head">
                  <p>{sectionName(activeDetail.section)} / {activeDetail.category || 'Uncategorized'}</p>
                  <h2>{activeDetail.title}</h2>
                  <div className="wiki-atlas__reader-meta">
                    <span>{activeDetail.source_count} sources</span>
                    <span>{activeDetail.backlink_count} backlinks</span>
                    <span>Updated {displayDate(activeDetail.last_updated)}</span>
                  </div>
                </div>
                <div className="wiki-atlas__markdown">
                  <ReactMarkdown components={markdownComponents}>
                    {activeDetail.render_markdown}
                  </ReactMarkdown>
                </div>
              </>
            )}
          </article>

          <aside className="wiki-atlas__provenance" aria-label="Source trail and related pages">
            {activeDetail ? (
              <>
                <section>
                  <h2>Source Trail</h2>
                  <dl className="wiki-atlas__facts">
                    <div>
                      <dt>Path</dt>
                      <dd>{activeDetail.path}</dd>
                    </div>
                    <div>
                      <dt>Confidence</dt>
                      <dd>{activeDetail.confidence || 'Unstated'}</dd>
                    </div>
                    <div>
                      <dt>Updated</dt>
                      <dd>{displayDate(activeDetail.last_updated)}</dd>
                    </div>
                  </dl>

                  {activeDetail.sources.length > 0 ? (
                    <ul className="wiki-atlas__source-list">
                      {activeDetail.sources.map((source) => (
                        <li key={source}>{source}</li>
                      ))}
                    </ul>
                  ) : (
                    <p className="wiki-atlas__muted">No explicit source strings are attached to this page.</p>
                  )}

                  {activeDetail.quality_flags.length > 0 && (
                    <div className="wiki-atlas__quality">
                      {activeDetail.quality_flags.map((flag) => (
                        <span key={flag}>{qualityLabel(flag)}</span>
                      ))}
                    </div>
                  )}
                </section>

                <section>
                  <h2>Outgoing Links</h2>
                  <div className="wiki-atlas__link-stack">
                    {activeDetail.outgoing_links.length === 0 && <p className="wiki-atlas__muted">No outgoing links.</p>}
                    {activeDetail.outgoing_links.map((link) => (
                      link.resolved ? (
                        <button key={`${link.target}-${link.url}`} type="button" onClick={() => navigateTo(link.url)}>
                          {link.title || link.label}
                        </button>
                      ) : (
                        <span key={link.target}>{link.label}</span>
                      )
                    ))}
                  </div>
                </section>

                <section>
                  <h2>Backlinks</h2>
                  <div className="wiki-atlas__link-stack">
                    {activeDetail.backlinks.length === 0 && <p className="wiki-atlas__muted">No backlinks yet.</p>}
                    {activeDetail.backlinks.map((page) => (
                      <button key={page.id} type="button" onClick={() => openPage(page)}>
                        {page.title}
                      </button>
                    ))}
                  </div>
                </section>
              </>
            ) : (
              <div className="wiki-atlas__reader-empty">
                <h2>Source Trail</h2>
                <p>Open a page to inspect sources, confidence, backlinks, and outgoing links.</p>
              </div>
            )}
          </aside>
        </main>
      )}
    </div>
  );
}
