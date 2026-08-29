import { useState, useCallback, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader } from 'lucide-react'
import { uploadDocument, getIngestionStatus } from '../services/api'
import type { IngestionStatusResponse } from '../services/api'

// ── Pipeline stages metadata ──────────────────────────────────────────────────
const STAGES = [
  { key: 'EXTRACTION',           label: 'PDF Validation & Extraction', desc: 'Validating structure and extracting text, tables, images' },
  { key: 'CHUNKING',             label: 'Document Chunking',           desc: 'Splitting content into semantic chunks' },
  { key: 'NORMALIZATION',        label: 'Chunk Normalization',         desc: 'Cleaning and normalizing chunk whitespace' },
  { key: 'VALIDATION',           label: 'Chunk Validation',           desc: 'Validating chunk integrity and contracts' },
  { key: 'EMBEDDING_PREPARATION',label: 'Embedding Preparation',      desc: 'Preparing records for vector embedding' },
  { key: 'EMBEDDING_GENERATION', label: 'Embedding Generation',       desc: 'Generating dense embedding vectors' },
  { key: 'COMPLETED',            label: 'Vector Storage',             desc: 'Storing vectors in Qdrant' },
]

type StageState = 'done' | 'active' | 'failed' | 'pending'

function stageState(stageKey: string, status: IngestionStatusResponse): StageState {
  if (status.status === 'FAILED' && status.current_stage === stageKey) return 'failed'
  if (status.completed_stages.includes(stageKey)) return 'done'
  if (status.status === 'COMPLETED') return 'done'
  if (status.current_stage === stageKey && status.status === 'RUNNING') return 'active'
  return 'pending'
}

// ── Stage indicator ───────────────────────────────────────────────────────────
function StageIndicator({ stage, state, index }: {
  stage: typeof STAGES[0]; state: StageState; index: number
}) {
  const dotClass = `stage-dot stage-dot-${state}`
  return (
    <div className="stage-item">
      <motion.div
        className={dotClass}
        initial={false}
        animate={state === 'active' ? { scale: [1, 1.15, 1] } : { scale: 1 }}
        transition={{ repeat: state === 'active' ? Infinity : 0, duration: 1.2 }}
      >
        {state === 'done' ? '✓' :
         state === 'failed' ? '✕' :
         state === 'active' ? <Loader size={12} style={{ animation: 'spin .7s linear infinite' }} /> :
         index + 1}
      </motion.div>
      <div className="stage-content">
        <div className="stage-name" style={{ color: state === 'done' ? 'var(--color-success)' : state === 'active' ? 'var(--color-purple)' : state === 'failed' ? 'var(--color-error)' : undefined }}>
          {stage.label}
        </div>
        <div className="stage-desc">{stage.desc}</div>
      </div>
    </div>
  )
}

export default function UploadPage() {
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [documentId, setDocumentId] = useState<string | null>(null)
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatusResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Poll ingestion status
  useEffect(() => {
    if (!documentId) return
    const poll = async () => {
      try {
        const status = await getIngestionStatus(documentId)
        setIngestionStatus(status)
        if (status.status === 'COMPLETED' || status.status === 'FAILED') {
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch {
        // status not available yet, retry
      }
    }
    poll()
    pollRef.current = setInterval(poll, 1500)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [documentId])

  const validateFile = (file: File): string | null => {
    if (!file.name.toLowerCase().endsWith('.pdf')) return 'Only PDF files are accepted.'
    if (file.size === 0) return 'File is empty.'
    if (file.size > 50 * 1024 * 1024) return 'File exceeds 50 MB limit.'
    return null
  }

  const handleFile = (file: File) => {
    const err = validateFile(file)
    if (err) { setError(err); return }
    setError(null)
    setSelectedFile(file)
    setDocumentId(null)
    setIngestionStatus(null)
    setUploadProgress(0)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }, [])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setUploading(true); setError(null); setUploadProgress(0)
    try {
      const res = await uploadDocument(selectedFile, setUploadProgress)
      setDocumentId(res.document_id)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const ingestionDone = ingestionStatus?.status === 'COMPLETED'
  const ingestionFailed = ingestionStatus?.status === 'FAILED'

  return (
    <>
      <div className="page-header">
        <motion.h1 className="page-title" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          Upload PDF Document
        </motion.h1>
        <p className="page-subtitle">Upload a PDF to trigger the multi-modal ingestion pipeline</p>
      </div>

      <div className="page-body" style={{ maxWidth: 820 }}>
        {/* Drop zone */}
        {!documentId && (
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
            <div
              className={`drop-zone ${dragging ? 'dragging' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input ref={fileInputRef} type="file" accept=".pdf" style={{ display: 'none' }} onChange={onInputChange} />
              <Upload className="drop-zone-icon" />
              <div className="drop-zone-title">Drag & drop your PDF here</div>
              <div className="drop-zone-subtitle">or click to browse files (max 50 MB)</div>
            </div>

            {/* Selected file info */}
            <AnimatePresence>
              {selectedFile && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ marginTop: 16 }}
                >
                  <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <FileText size={24} color="var(--color-purple)" />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 14 }}>{selectedFile.name}</div>
                      <div style={{ fontSize: 12, color: 'var(--color-gray-500)' }}>
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </div>
                    </div>
                    <button className="btn btn-ghost" onClick={e => { e.stopPropagation(); setSelectedFile(null) }}>
                      <X size={15} />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Error */}
            {error && (
              <motion.div className="alert alert-error" style={{ marginTop: 12 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <AlertCircle size={16} /> {error}
              </motion.div>
            )}

            {/* Upload button + progress */}
            {selectedFile && (
              <div style={{ marginTop: 16 }}>
                {uploading ? (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 13, fontWeight: 500 }}>
                      <span>Uploading…</span><span>{uploadProgress}%</span>
                    </div>
                    <div className="progress-track">
                      <motion.div
                        className="progress-fill"
                        animate={{ width: `${uploadProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                ) : (
                  <button className="btn btn-primary" onClick={handleUpload} disabled={uploading}>
                    <Upload size={14} /> Start Ingestion
                  </button>
                )}
              </div>
            )}
          </motion.div>
        )}

        {/* Ingestion progress panel */}
        {documentId && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            {/* Status header */}
            <div className="card" style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                {ingestionFailed ? <AlertCircle size={22} color="var(--color-error)" /> :
                 ingestionDone ? <CheckCircle size={22} color="var(--color-success)" /> :
                 <div className="spinner" />}
                <div>
                  <div style={{ fontWeight: 700, fontSize: 15 }}>
                    {ingestionDone ? 'Ingestion Complete!' :
                     ingestionFailed ? 'Ingestion Failed' :
                     'Processing…'}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--color-gray-500)', marginTop: 2 }}>
                    Document ID: <code style={{ fontSize: 11 }}>{documentId}</code>
                  </div>
                </div>
                <div style={{ marginLeft: 'auto' }}>
                  {ingestionStatus && (
                    <span className={`badge ${ingestionDone ? 'badge-success' : ingestionFailed ? 'badge-error' : 'badge-info'}`}>
                      {ingestionStatus.progress}%
                    </span>
                  )}
                </div>
              </div>

              {ingestionStatus && (
                <div style={{ marginTop: 14 }}>
                  <div className="progress-track">
                    <motion.div
                      className={`progress-fill ${ingestionDone ? 'progress-fill-success' : ingestionFailed ? 'progress-fill-error' : ''}`}
                      animate={{ width: `${ingestionStatus.progress}%` }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                  {ingestionStatus.errors.length > 0 && (
                    <div className="alert alert-error" style={{ marginTop: 12 }}>
                      <AlertCircle size={14} /> {ingestionStatus.errors[0]}
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Metrics */}
            {ingestionDone && ingestionStatus && (
              <motion.div
                className="card"
                style={{ marginBottom: 16, display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              >
                {[
                  { label: 'Total Chunks', val: ingestionStatus.chunks },
                  { label: 'Text Chunks', val: ingestionStatus.text_chunks },
                  { label: 'Table Chunks', val: ingestionStatus.table_chunks },
                  { label: 'Image Chunks', val: ingestionStatus.image_chunks },
                ].map(({ label, val }) => (
                  <div key={label}>
                    <div className="card-title">{label}</div>
                    <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--color-black)' }}>{val}</div>
                  </div>
                ))}
              </motion.div>
            )}

            {/* Pipeline stages */}
            <div className="card">
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 20 }}>Pipeline Stages</div>
              <div className="stage-list">
                {STAGES.map((stage, i) => (
                  <StageIndicator
                    key={stage.key}
                    stage={stage}
                    index={i}
                    state={ingestionStatus ? stageState(stage.key, ingestionStatus) : 'pending'}
                  />
                ))}
              </div>
            </div>

            {/* CTA after completion */}
            {ingestionDone && (
              <motion.div style={{ marginTop: 16, display: 'flex', gap: 12 }} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <button className="btn btn-primary" onClick={() => navigate(`/documents/${documentId}`)}>
                  View Document
                </button>
                <button className="btn btn-secondary" onClick={() => navigate('/search')}>
                  Search Documents
                </button>
                <button className="btn btn-ghost" onClick={() => { setDocumentId(null); setSelectedFile(null); setIngestionStatus(null) }}>
                  Upload Another
                </button>
              </motion.div>
            )}
          </motion.div>
        )}
      </div>
    </>
  )
}
