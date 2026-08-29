import { useEffect, useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, RefreshCw, Search, Layers, Clock, AlertCircle, CheckCircle } from 'lucide-react'
import { getDocument, getIngestionStatus } from '../services/api'
import type { DocumentDetailResponse, IngestionStatusResponse } from '../services/api'

type Tab = 'overview' | 'chunks' | 'metrics' | 'logs'

const PIPELINE_STAGES = [
  'EXTRACTION', 'CHUNKING', 'NORMALIZATION', 'VALIDATION',
  'EMBEDDING_PREPARATION', 'EMBEDDING_GENERATION', 'COMPLETED',
]

function Badge({ status }: { status: string }) {
  const cls =
    status === 'completed' ? 'badge badge-success' :
    status === 'processing' ? 'badge badge-info' :
    status === 'error' ? 'badge badge-error' :
    status === 'COMPLETED' ? 'badge badge-success' :
    status === 'RUNNING' ? 'badge badge-info' :
    status === 'FAILED' ? 'badge badge-error' :
    'badge badge-neutral'
  return <span className={cls}>{status}</span>
}

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [doc, setDoc] = useState<DocumentDetailResponse | null>(null)
  const [ingestion, setIngestion] = useState<IngestionStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('overview')

  const load = async () => {
    if (!id) return
    setLoading(true)
    try {
      const [docRes, ingRes] = await Promise.all([
        getDocument(id),
        getIngestionStatus(id).catch(() => null),
      ])
      setDoc(docRes)
      setIngestion(ingRes)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load document')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [id])

  if (loading) {
    return (
      <div className="page-body">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxWidth: 900 }}>
          {[1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 80 }} />)}
        </div>
      </div>
    )
  }

  if (error || !doc) {
    return (
      <div className="page-body">
        <div className="alert alert-error">
          <AlertCircle size={16} /> {error ?? 'Document not found'}
        </div>
        <button className="btn btn-secondary" style={{ marginTop: 12 }} onClick={() => navigate(-1)}>
          <ArrowLeft size={13} /> Back
        </button>
      </div>
    )
  }

  const effIngestion = ingestion ?? doc.ingestion_status

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <button className="btn btn-ghost" onClick={() => navigate('/documents')} style={{ padding: '5px 8px' }}>
            <ArrowLeft size={15} />
          </button>
          <motion.h1 className="page-title" style={{ margin: 0 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {doc.filename}
          </motion.h1>
          <Badge status={doc.status} />
        </div>
        <p className="page-subtitle" style={{ fontFamily: 'monospace', fontSize: 11 }}>{doc.document_id}</p>
      </div>

      <div className="page-body" style={{ maxWidth: 940 }}>
        {/* Tabs */}
        <div className="tab-list">
          {(['overview', 'chunks', 'metrics', 'logs'] as Tab[]).map(tab => (
            <div
              key={tab}
              className={`tab-item ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </div>
          ))}
        </div>

        {/* Overview tab */}
        {activeTab === 'overview' && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
              {/* Document info */}
              <div className="card">
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>Document Info</div>
                {[
                  ['Filename', doc.filename],
                  ['Document ID', doc.document_id],
                  ['File Size', `${(doc.file_size_bytes / 1024).toFixed(1)} KB`],
                  ['Uploaded', new Date(doc.upload_time).toLocaleString()],
                  ['Status', doc.status],
                ].map(([label, value]) => (
                  <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--color-border)', fontSize: 13 }}>
                    <span style={{ color: 'var(--color-gray-500)', fontWeight: 500 }}>{label}</span>
                    <span style={{ fontWeight: 600, maxWidth: '60%', textAlign: 'right', wordBreak: 'break-all', fontSize: 12 }}>{value}</span>
                  </div>
                ))}
              </div>

              {/* Ingestion summary */}
              <div className="card">
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>Ingestion Summary</div>
                {effIngestion ? (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                      {effIngestion.status === 'COMPLETED' ? <CheckCircle size={18} color="var(--color-success)" /> :
                       effIngestion.status === 'FAILED' ? <AlertCircle size={18} color="var(--color-error)" /> :
                       <div className="spinner" />}
                      <Badge status={effIngestion.status} />
                    </div>
                    <div className="progress-track" style={{ marginBottom: 14 }}>
                      <div
                        className={`progress-fill ${effIngestion.status === 'COMPLETED' ? 'progress-fill-success' : effIngestion.status === 'FAILED' ? 'progress-fill-error' : ''}`}
                        style={{ width: `${effIngestion.progress}%` }}
                      />
                    </div>
                    {[
                      ['Current Stage', effIngestion.current_stage],
                      ['Progress', `${effIngestion.progress}%`],
                      ['Duration', `${effIngestion.duration_seconds.toFixed(2)}s`],
                    ].map(([label, value]) => (
                      <div key={label} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0', borderBottom: '1px solid var(--color-border)', fontSize: 13 }}>
                        <span style={{ color: 'var(--color-gray-500)', fontWeight: 500 }}>{label}</span>
                        <span style={{ fontWeight: 600 }}>{value}</span>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="empty-state" style={{ padding: '20px 0' }}>
                    <Clock size={28} color="var(--color-gray-300)" style={{ margin: '0 auto 8px' }} />
                    <div style={{ fontSize: 13, color: 'var(--color-gray-500)' }}>No ingestion data yet</div>
                  </div>
                )}
              </div>
            </div>

            {/* Pipeline stages */}
            {effIngestion && (
              <div className="card">
                <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 18 }}>Pipeline Progress</div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
                  {PIPELINE_STAGES.map(stage => {
                    const done = effIngestion.completed_stages.includes(stage) || effIngestion.status === 'COMPLETED'
                    const active = effIngestion.current_stage === stage && effIngestion.status === 'RUNNING'
                    const failed = effIngestion.status === 'FAILED' && effIngestion.current_stage === stage
                    return (
                      <div key={stage} style={{
                        padding: '10px 12px', borderRadius: 8,
                        border: `1.5px solid ${done ? 'var(--color-success)' : active ? 'var(--color-purple)' : failed ? 'var(--color-error)' : 'var(--color-border)'}`,
                        background: done ? '#f0fdf4' : active ? 'var(--color-purple-lt)' : 'var(--color-bg)',
                      }}>
                        <div style={{ fontSize: 10, fontWeight: 700, color: done ? 'var(--color-success)' : active ? 'var(--color-purple)' : failed ? 'var(--color-error)' : 'var(--color-gray-500)', textTransform: 'uppercase', letterSpacing: '.4px' }}>
                          {done ? '✓ Done' : active ? '⟳ Active' : failed ? '✕ Failed' : 'Pending'}
                        </div>
                        <div style={{ fontSize: 12, fontWeight: 600, marginTop: 3 }}>{stage.replace(/_/g, ' ')}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Chunks tab */}
        {activeTab === 'chunks' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 20 }}>
              {[
                { label: 'Total', val: doc.total_chunks, color: 'var(--color-purple)' },
                { label: 'Text', val: effIngestion?.text_chunks ?? 0, color: '#0891b2' },
                { label: 'Table', val: effIngestion?.table_chunks ?? 0, color: '#d97706' },
                { label: 'Image', val: effIngestion?.image_chunks ?? 0, color: '#7c3aed' },
              ].map(({ label, val, color }) => (
                <div key={label} className="card">
                  <div className="card-title">{label} Chunks</div>
                  <div style={{ fontSize: 28, fontWeight: 800, color }}>{val}</div>
                </div>
              ))}
            </div>
            <div className="card">
              <div className="empty-state" style={{ padding: '32px 0' }}>
                <Layers size={36} color="var(--color-gray-300)" style={{ margin: '0 auto 10px' }} />
                <div className="empty-title">Chunk browser coming soon</div>
                <div className="empty-desc">Individual chunk content will be displayed here.</div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Metrics tab */}
        {activeTab === 'metrics' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            {effIngestion?.stage_metrics && effIngestion.stage_metrics.length > 0 ? (
              <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Stage</th>
                      <th>Duration (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {effIngestion.stage_metrics.map(sm => (
                      <tr key={sm.stage}>
                        <td style={{ fontWeight: 600 }}>{sm.stage}</td>
                        <td>{sm.duration_seconds?.toFixed(3) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="card">
                <div className="empty-state" style={{ padding: '32px 0' }}>
                  <Clock size={36} color="var(--color-gray-300)" style={{ margin: '0 auto 10px' }} />
                  <div className="empty-title">No metrics available</div>
                  <div className="empty-desc">Metrics are recorded during ingestion.</div>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Logs tab */}
        {activeTab === 'logs' && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="card">
              {effIngestion?.errors && effIngestion.errors.length > 0 ? (
                <div>
                  <div style={{ fontWeight: 700, marginBottom: 14 }}>Errors</div>
                  {effIngestion.errors.map((err, i) => (
                    <div key={i} className="alert alert-error" style={{ marginBottom: 8 }}>
                      <AlertCircle size={14} /> {err}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state" style={{ padding: '32px 0' }}>
                  <CheckCircle size={36} color="var(--color-gray-300)" style={{ margin: '0 auto 10px' }} />
                  <div className="empty-title">No errors logged</div>
                  <div className="empty-desc">
                    {effIngestion?.status === 'COMPLETED' ? 'Pipeline completed successfully.' : 'Logs will appear here if errors occur.'}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}

        {/* Actions footer */}
        <div style={{ marginTop: 24, display: 'flex', gap: 12 }}>
          <Link to="/search" className="btn btn-primary">
            <Search size={13} /> Search This Document
          </Link>
          <button className="btn btn-secondary" onClick={load}>
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>
    </>
  )
}
