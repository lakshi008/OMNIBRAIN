import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { FileText, Search, Trash2, RefreshCw, Upload, Eye } from 'lucide-react'
import { getDocuments, deleteDocument, reingestDocument } from '../services/api'
import type { DocumentRecord } from '../services/api'

function StatusBadge({ status }: { status: string }) {
  const cls =
    status === 'completed' ? 'badge badge-success' :
    status === 'processing' ? 'badge badge-info' :
    status === 'error' ? 'badge badge-error' :
    'badge badge-neutral'
  return <span className={cls}>{status}</span>
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default function DocumentsPage() {
  const navigate = useNavigate()
  const [docs, setDocs] = useState<DocumentRecord[]>([])
  const [filtered, setFiltered] = useState<DocumentRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [error, setError] = useState<string | null>(null)
  const [actionId, setActionId] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const res = await getDocuments()
      setDocs(res.documents ?? [])
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load documents')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  useEffect(() => {
    let result = docs
    if (statusFilter !== 'all') result = result.filter(d => d.status === statusFilter)
    if (search.trim()) result = result.filter(d =>
      d.filename.toLowerCase().includes(search.toLowerCase()) ||
      d.document_id.toLowerCase().includes(search.toLowerCase())
    )
    setFiltered(result)
  }, [docs, search, statusFilter])

  const handleDelete = async (docId: string) => {
    if (!confirm('Remove this document from the registry?')) return
    setActionId(docId)
    try {
      await deleteDocument(docId)
      setDocs(prev => prev.filter(d => d.document_id !== docId))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Delete failed')
    } finally {
      setActionId(null)
    }
  }

  const handleReingest = async (docId: string) => {
    setActionId(docId)
    try {
      await reingestDocument(docId)
      await load()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Re-ingestion failed')
    } finally {
      setActionId(null)
    }
  }

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              Documents
            </motion.h1>
            <p className="page-subtitle">{docs.length} document{docs.length !== 1 ? 's' : ''} in registry</p>
          </div>
          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn btn-secondary" onClick={load}>
              <RefreshCw size={13} /> Refresh
            </button>
            <Link to="/documents/upload" className="btn btn-primary">
              <Upload size={13} /> Upload
            </Link>
          </div>
        </div>
      </div>

      <div className="page-body">
        {error && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            {error}
            <button style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'inherit' }} onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Filters */}
        <motion.div
          className="card"
          style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--color-gray-500)' }} />
            <input
              type="text"
              placeholder="Search by filename or ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ width: '100%', paddingLeft: 32 }}
            />
          </div>
          <select value={statusFilter} onChange={e => setStatusFilter(e.target.value)}>
            <option value="all">All statuses</option>
            <option value="completed">Completed</option>
            <option value="processing">Processing</option>
            <option value="error">Error</option>
            <option value="uploaded">Uploaded</option>
          </select>
        </motion.div>

        {/* Table */}
        <motion.div className="card" style={{ padding: 0, overflow: 'hidden' }} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          {loading ? (
            <div style={{ padding: 32, display: 'flex', flexDirection: 'column', gap: 14 }}>
              {[1, 2, 3, 4].map(i => <div key={i} className="skeleton" style={{ height: 48 }} />)}
            </div>
          ) : filtered.length === 0 ? (
            <div className="empty-state">
              <FileText size={40} color="var(--color-gray-300)" style={{ margin: '0 auto 14px' }} />
              <div className="empty-title">No documents found</div>
              <div className="empty-desc">
                {docs.length === 0 ? 'Upload a PDF to get started.' : 'Try adjusting your search or filters.'}
              </div>
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Status</th>
                  <th>Chunks</th>
                  <th>Size</th>
                  <th>Uploaded</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((doc, idx) => (
                  <motion.tr
                    key={doc.document_id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.03 }}
                  >
                    <td>
                      <div style={{ fontWeight: 600, fontSize: 13 }}>{doc.filename}</div>
                      <div style={{ fontSize: 10, color: 'var(--color-gray-500)', fontFamily: 'monospace', marginTop: 2 }}>
                        {doc.document_id}
                      </div>
                    </td>
                    <td><StatusBadge status={doc.status} /></td>
                    <td style={{ fontWeight: 600 }}>{doc.total_chunks}</td>
                    <td style={{ fontSize: 12.5, color: 'var(--color-gray-500)' }}>{formatBytes(doc.file_size_bytes)}</td>
                    <td style={{ fontSize: 12, color: 'var(--color-gray-500)' }}>
                      {new Date(doc.upload_time).toLocaleDateString()}
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button
                          className="btn btn-ghost"
                          style={{ padding: '5px 8px' }}
                          title="View details"
                          onClick={() => navigate(`/documents/${doc.document_id}`)}
                        >
                          <Eye size={13} />
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ padding: '5px 8px' }}
                          title="Re-ingest"
                          disabled={actionId === doc.document_id}
                          onClick={() => handleReingest(doc.document_id)}
                        >
                          <RefreshCw size={13} />
                        </button>
                        <button
                          className="btn btn-ghost"
                          style={{ padding: '5px 8px', color: 'var(--color-error)' }}
                          title="Delete"
                          disabled={actionId === doc.document_id}
                          onClick={() => handleDelete(doc.document_id)}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          )}
        </motion.div>
      </div>
    </>
  )
}
