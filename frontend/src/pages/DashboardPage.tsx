import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  FileText, CheckCircle, Clock, AlertCircle,
  Upload, Search, Layers,
} from 'lucide-react'
import {
  getDocuments,
  getHealth,
} from '../services/api'
import type {
  DocumentRecord,
  HealthResponse,
} from '../services/api'

// ── Animated counter ──────────────────────────────────────────────────────────
function Counter({ value, duration = 800 }: { value: number; duration?: number }) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const tick = () => {
      const elapsed = Date.now() - start
      const progress = Math.min(elapsed / duration, 1)
      const ease = 1 - Math.pow(1 - progress, 3)
      setDisplay(Math.round(ease * value))
      if (progress < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value, duration])
  return <>{display.toLocaleString()}</>
}

// ── Stat card ─────────────────────────────────────────────────────────────────
function StatCard({
  label, value, icon: Icon, color, delay,
}: {
  label: string; value: number; icon: React.ElementType; color: string; delay: number
}) {
  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <div className="card-title">{label}</div>
          <div className="stat-value"><Counter value={value} /></div>
        </div>
        <div style={{
          width: 44, height: 44, borderRadius: 10,
          background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Icon size={20} color="#fff" />
        </div>
      </div>
    </motion.div>
  )
}

// ── Status badge ──────────────────────────────────────────────────────────────
function DocStatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    completed: 'badge badge-success',
    processing: 'badge badge-info',
    error: 'badge badge-error',
    uploaded: 'badge badge-neutral',
  }
  return <span className={map[status] ?? 'badge badge-neutral'}>{status}</span>
}

export default function DashboardPage() {
  const [docs, setDocs] = useState<DocumentRecord[]>([])
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getDocuments().catch(() => ({ documents: [], total: 0 })),
      getHealth().catch(() => null),
    ]).then(([docsRes, healthRes]) => {
      setDocs(docsRes.documents ?? [])
      setHealth(healthRes)
      setLoading(false)
    })
  }, [])

  const total = docs.length
  const completed = docs.filter(d => d.status === 'completed').length
  const processing = docs.filter(d => d.status === 'processing').length
  const failed = docs.filter(d => d.status === 'error').length
  const totalChunks = docs.reduce((s, d) => s + d.total_chunks, 0)
  const recent = docs.slice(0, 6)

  return (
    <>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <motion.h1
              className="page-title"
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
            >
              Dashboard
            </motion.h1>
            <p className="page-subtitle">Overview of your OMNIBRAIN document intelligence system</p>
          </div>
          <Link to="/documents/upload" className="btn btn-primary">
            <Upload size={14} /> Upload PDF
          </Link>
        </div>
      </div>

      <div className="page-body">
        {/* Stats grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 28 }}>
          <StatCard label="Total Documents" value={total} icon={FileText} color="#7c3aed" delay={0} />
          <StatCard label="Processed" value={completed} icon={CheckCircle} color="#16a34a" delay={0.05} />
          <StatCard label="Processing" value={processing} icon={Clock} color="#d97706" delay={0.1} />
          <StatCard label="Failed" value={failed} icon={AlertCircle} color="#dc2626" delay={0.15} />
          <StatCard label="Total Chunks" value={totalChunks} icon={Layers} color="#0891b2" delay={0.2} />
        </div>

        {/* Main grid: recent docs + health + quick actions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20 }}>
          {/* Recent documents */}
          <motion.div
            className="card"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
              <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>Recent Documents</h2>
              <Link to="/documents" style={{ fontSize: 12, color: 'var(--color-purple)', fontWeight: 600, textDecoration: 'none' }}>
                View all →
              </Link>
            </div>
            {loading ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {[1, 2, 3].map(i => (
                  <div key={i} className="skeleton" style={{ height: 44 }} />
                ))}
              </div>
            ) : recent.length === 0 ? (
              <div className="empty-state" style={{ padding: '32px 0' }}>
                <FileText size={36} color="var(--color-gray-300)" style={{ margin: '0 auto 12px' }} />
                <div className="empty-title">No documents yet</div>
                <div className="empty-desc">Upload your first PDF to get started.</div>
              </div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Status</th>
                    <th>Chunks</th>
                    <th>Upload Time</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map(doc => (
                    <tr key={doc.document_id}>
                      <td>
                        <Link
                          to={`/documents/${doc.document_id}`}
                          style={{ color: 'var(--color-purple)', fontWeight: 600, textDecoration: 'none', fontSize: 13 }}
                        >
                          {doc.filename}
                        </Link>
                        <div style={{ fontSize: 11, color: 'var(--color-gray-500)', marginTop: 2, fontFamily: 'monospace' }}>
                          {doc.document_id.slice(0, 12)}…
                        </div>
                      </td>
                      <td><DocStatusBadge status={doc.status} /></td>
                      <td style={{ fontWeight: 600 }}>{doc.total_chunks}</td>
                      <td style={{ fontSize: 12, color: 'var(--color-gray-500)' }}>
                        {new Date(doc.upload_time).toLocaleString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </motion.div>

          {/* Right column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* System health */}
            <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <h2 style={{ margin: 0, fontSize: 15, fontWeight: 700 }}>System Health</h2>
                <Link to="/health" style={{ fontSize: 12, color: 'var(--color-purple)', fontWeight: 600, textDecoration: 'none' }}>View →</Link>
              </div>
              {health === null ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--color-gray-500)', fontSize: 13 }}>
                  <div className="health-dot health-dot-unknown" />
                  Backend unavailable
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {health.components.map(c => (
                    <div key={c.name} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div className={`health-dot health-dot-${c.status === 'healthy' ? 'healthy' : c.status === 'degraded' ? 'degraded' : 'unhealthy'}`} />
                      <span style={{ fontSize: 13, fontWeight: 500, flex: 1 }}>
                        {c.name.replace(/_/g, ' ')}
                      </span>
                      <span style={{ fontSize: 11, color: 'var(--color-gray-500)' }}>
                        {c.status}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>

            {/* Quick actions */}
            <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.35 }}>
              <h2 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 700 }}>Quick Actions</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <Link to="/documents/upload" className="btn btn-primary" style={{ justifyContent: 'flex-start' }}>
                  <Upload size={14} /> Upload PDF
                </Link>
                <Link to="/search" className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                  <Search size={14} /> Search Documents
                </Link>
                <Link to="/documents" className="btn btn-secondary" style={{ justifyContent: 'flex-start' }}>
                  <FileText size={14} /> Manage Documents
                </Link>
              </div>
            </motion.div>

            {/* Uptime */}
            {health && (
              <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <div className="card-title">API Uptime</div>
                <div className="stat-value" style={{ fontSize: 24 }}>
                  {Math.floor(health.uptime_seconds / 60)}m {Math.floor(health.uptime_seconds % 60)}s
                </div>
                <div className="stat-label">v{health.version} · {health.status}</div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
