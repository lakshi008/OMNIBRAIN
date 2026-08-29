import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, Loader, AlertCircle, FileText, ChevronDown, ChevronUp, Cpu, Sparkles, BookOpen } from 'lucide-react'
import { searchDocuments } from '../services/api'
import type { SearchResponse, CitationItem } from '../services/api'

// ── Citation card ─────────────────────────────────────────────────────────────
function CitationCard({ citation, index }: { citation: CitationItem; index: number }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <motion.div
      className="citation-card"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
    >
      <div className="citation-header">
        <span className="citation-label">[{index + 1}] {citation.content_type}</span>
        <span className="citation-score">score: {citation.score.toFixed(3)}</span>
      </div>
      <div className="citation-source">
        <FileText size={13} style={{ display: 'inline', marginRight: 6, color: 'var(--color-purple)', verticalAlign: 'middle' }} />
        {citation.citation}
      </div>
      {citation.content && (
        <>
          <div className="citation-content" style={{ maxHeight: expanded ? 'none' : 80 }}>
            {citation.content}
          </div>
          {citation.content.length > 200 && (
            <button
              onClick={() => setExpanded(!expanded)}
              style={{ background: 'none', border: 'none', color: 'var(--color-purple)', fontSize: 12, fontWeight: 600, cursor: 'pointer', marginTop: 6, display: 'flex', alignItems: 'center', gap: 4 }}
            >
              {expanded ? <><ChevronUp size={12} /> Show less</> : <><ChevronDown size={12} /> Show more</>}
            </button>
          )}
        </>
      )}
    </motion.div>
  )
}

// ── Skeleton ──────────────────────────────────────────────────────────────────
function SearchSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {[1, 2, 3].map(i => (
        <div key={i} className="skeleton" style={{ height: 120, borderRadius: 10 }} />
      ))}
    </div>
  )
}

export default function SearchPage() {
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    const q = query.trim()
    if (!q) return
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await searchDocuments({ query: q, top_k: topK })
      setResult(res)
      if (res.error) setError(res.error)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSearch() }
  }

  return (
    <>
      <div className="page-header">
        <motion.div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <Cpu size={22} color="var(--color-purple)" />
          <h1 className="page-title" style={{ margin: 0 }}>AI Search & Synthesis</h1>
        </motion.div>
        <p className="page-subtitle">Ask questions about your documents — grounded with vector retrieval and citation provenance</p>
      </div>

      <div className="page-body" style={{ maxWidth: 860 }}>
        {/* Search input */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="search-box" style={{ marginBottom: 14 }}>
            <input
              className="search-input"
              placeholder="Ask anything about your documents…"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKey}
            />
            <button
              className="search-btn"
              onClick={handleSearch}
              disabled={loading || !query.trim()}
            >
              {loading ? <Loader size={15} style={{ animation: 'spin .7s linear infinite' }} /> : <Search size={15} />}
              Search
            </button>
          </div>

          {/* Options row */}
          <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 24 }}>
            <label style={{ fontSize: 13, color: 'var(--color-gray-500)', display: 'flex', alignItems: 'center', gap: 8 }}>
              Top K results:
              <input
                type="number"
                min={1} max={20}
                value={topK}
                onChange={e => setTopK(Number(e.target.value))}
                style={{ width: 60 }}
              />
            </label>
            <div style={{ fontSize: 12, color: 'var(--color-gray-500)' }}>
              Press <kbd style={{ background: 'var(--color-bg-subtle)', border: '1px solid var(--color-border)', borderRadius: 4, padding: '1px 6px', fontSize: 11 }}>Enter</kbd> to search
            </div>
          </div>
        </motion.div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div className="alert alert-error" style={{ marginBottom: 20 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <AlertCircle size={15} /> {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading */}
        {loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20, color: 'var(--color-purple)', fontSize: 14, fontWeight: 500 }}>
              <div className="spinner" /> Searching and synthesizing grounded answer…
            </div>
            <SearchSkeleton />
          </motion.div>
        )}

        {/* Results */}
        {result && !loading && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {/* Generated Answer Card */}
            {result.answer && (
              <motion.div
                className="card"
                style={{
                  marginBottom: 24,
                  border: '1.5px solid var(--color-purple)',
                  background: 'linear-gradient(180deg, #ffffff 0%, #faf8ff 100%)',
                  boxShadow: '0 4px 20px -2px rgba(124, 58, 237, 0.08)',
                }}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Sparkles size={18} color="var(--color-purple)" />
                    <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--color-purple)' }}>
                      Synthesized Answer
                    </span>
                  </div>
                  <span className="badge badge-info">
                    Grounded ({result.total_results} sources)
                  </span>
                </div>
                <div style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--color-black)', whiteSpace: 'pre-wrap' }}>
                  {result.answer}
                </div>
              </motion.div>
            )}

            {/* Summary bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <BookOpen size={16} color="var(--color-gray-500)" />
                <span style={{ fontWeight: 700, fontSize: 15 }}>
                  {result.total_results} Source Citation{result.total_results !== 1 ? 's' : ''}
                </span>
                <span style={{ fontSize: 13, color: 'var(--color-gray-500)' }}>
                  for "{result.query}"
                </span>
              </div>
              <span className={`badge ${result.status === 'RESULTS_FOUND' ? 'badge-success' : 'badge-neutral'}`}>
                {result.status.replace('_', ' ')}
              </span>
            </div>

            {/* No results */}
            {result.total_results === 0 && (
              <div className="empty-state">
                <Search size={40} color="var(--color-gray-300)" style={{ margin: '0 auto 14px' }} />
                <div className="empty-title">No results found</div>
                <div className="empty-desc">
                  Try a different query, or upload and ingest a document first.
                </div>
              </div>
            )}

            {/* Citations */}
            {result.results.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {result.results.map((citation, i) => (
                  <CitationCard key={citation.chunk_id} citation={citation} index={i} />
                ))}
              </div>
            )}

            {/* Context block (collapsible) */}
            {result.context && (
              <motion.details
                style={{ marginTop: 24 }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2 }}
              >
                <summary style={{ cursor: 'pointer', fontSize: 13, fontWeight: 600, color: 'var(--color-gray-500)', marginBottom: 10, listStyle: 'none', display: 'flex', alignItems: 'center', gap: 6 }}>
                  <ChevronDown size={13} /> View structured retrieval context passed to LLM
                </summary>
                <pre style={{
                  background: 'var(--color-bg-subtle)', border: '1px solid var(--color-border)',
                  borderRadius: 10, padding: 20, fontSize: 12, overflowX: 'auto',
                  color: 'var(--color-gray-700)', lineHeight: 1.7, fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap', maxHeight: 400, overflow: 'auto',
                }}>
                  {result.context}
                </pre>
              </motion.details>
            )}
          </motion.div>
        )}

        {/* Initial state */}
        {!result && !loading && !error && (
          <motion.div className="empty-state" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 0.2 } }}>
            <Search size={48} color="var(--color-purple-dim)" style={{ margin: '0 auto 16px' }} />
            <div className="empty-title">Ready to search & synthesize</div>
            <div className="empty-desc">
              Type a question above and press Search. Answers are synthesized directly from retrieved document evidence.
            </div>
          </motion.div>
        )}
      </div>
    </>
  )
}
