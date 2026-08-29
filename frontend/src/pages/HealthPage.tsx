import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Heart, RefreshCw, CheckCircle, AlertCircle, AlertTriangle } from 'lucide-react'
import { getHealth, getSearchHealth } from '../services/api'
import type { HealthResponse, SearchHealthResponse } from '../services/api'

type Status = 'healthy' | 'degraded' | 'unhealthy' | 'unknown'

function statusIcon(s: Status) {
  if (s === 'healthy') return <CheckCircle size={18} color="var(--color-success)" />
  if (s === 'degraded') return <AlertTriangle size={18} color="var(--color-warning)" />
  if (s === 'unhealthy') return <AlertCircle size={18} color="var(--color-error)" />
  return <div className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} />
}

function HealthRow({ name, status, message }: { name: string; status: string; message: string }) {
  const s = (status as Status) ?? 'unknown'
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      style={{
        display: 'flex', alignItems: 'center', gap: 14, padding: '14px 0',
        borderBottom: '1px solid var(--color-border)',
      }}
    >
      {statusIcon(s)}
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 14, fontWeight: 600 }}>{name.replace(/_/g, ' ')}</div>
        {message && <div style={{ fontSize: 12, color: 'var(--color-gray-500)', marginTop: 2 }}>{message}</div>}
      </div>
      <span className={`badge ${s === 'healthy' ? 'badge-success' : s === 'degraded' ? 'badge-warning' : s === 'unhealthy' ? 'badge-error' : 'badge-neutral'}`}>
        {s}
      </span>
    </motion.div>
  )
}

export default function HealthPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [searchHealth, setSearchHealth] = useState<SearchHealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true); setError(null)
    try {
      const [h, sh] = await Promise.all([
        getHealth().catch(() => null),
        getSearchHealth().catch(() => null),
      ])
      setHealth(h)
      setSearchHealth(sh)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Health check failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const overallStatus = health?.status ?? 'unknown'

  return (
    <>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <motion.div style={{ display: 'flex', alignItems: 'center', gap: 10 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Heart size={22} color="var(--color-purple)" />
              <h1 className="page-title" style={{ margin: 0 }}>System Health</h1>
            </motion.div>
            <p className="page-subtitle">Real-time status of all system components</p>
          </div>
          <button className="btn btn-secondary" onClick={load}>
            <RefreshCw size={13} /> Refresh
          </button>
        </div>
      </div>

      <div className="page-body" style={{ maxWidth: 760 }}>
        {error && (
          <div className="alert alert-error" style={{ marginBottom: 16 }}>
            <AlertCircle size={15} /> Cannot reach backend: {error}
          </div>
        )}

        {/* Overall status banner */}
        <motion.div
          className="card"
          style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 16 }}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div style={{
            width: 56, height: 56, borderRadius: 12,
            background: overallStatus === 'healthy' ? '#dcfce7' : overallStatus === 'degraded' ? '#fef3c7' : '#fee2e2',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {loading
              ? <div className="spinner" />
              : overallStatus === 'healthy'
              ? <CheckCircle size={26} color="var(--color-success)" />
              : overallStatus === 'degraded'
              ? <AlertTriangle size={26} color="var(--color-warning)" />
              : <AlertCircle size={26} color="var(--color-error)" />}
          </div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, textTransform: 'capitalize' }}>
              {overallStatus === 'unknown' ? 'Checking…' : `System ${overallStatus}`}
            </div>
            {health && (
              <div style={{ fontSize: 12.5, color: 'var(--color-gray-500)', marginTop: 3 }}>
                v{health.version} · Uptime: {Math.floor(health.uptime_seconds / 60)}m {Math.floor(health.uptime_seconds % 60)}s ·{' '}
                {health.total_documents} document{health.total_documents !== 1 ? 's' : ''}
              </div>
            )}
          </div>
        </motion.div>

        {/* API components */}
        <motion.div className="card" style={{ marginBottom: 16 }} initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>API Components</div>
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {[1, 2, 3].map(i => <div key={i} className="skeleton" style={{ height: 50 }} />)}
            </div>
          ) : health ? (
            health.components.map(c => (
              <HealthRow key={c.name} name={c.name} status={c.status} message={c.message} />
            ))
          ) : (
            <div className="alert alert-error">
              <AlertCircle size={14} /> Backend API is unreachable. Is uvicorn running?
            </div>
          )}
        </motion.div>

        {/* Search subsystem */}
        <motion.div className="card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>Search Subsystem</div>
          {loading ? (
            <div className="skeleton" style={{ height: 50 }} />
          ) : searchHealth ? (
            <>
              <HealthRow name="search_service" status={searchHealth.status} message={searchHealth.message} />
              <HealthRow name="vector_store" status={searchHealth.vector_store} message={searchHealth.collection_exists ? `Collection: ${searchHealth.collection_name}` : 'No collection yet'} />
              <HealthRow name="embedding_provider" status={searchHealth.embedding_provider} message={`Collection exists: ${searchHealth.collection_exists}`} />
            </>
          ) : (
            <HealthRow name="search_service" status="unknown" message="Could not reach search health endpoint" />
          )}
        </motion.div>
      </div>
    </>
  )
}
