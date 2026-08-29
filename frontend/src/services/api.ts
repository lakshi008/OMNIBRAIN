/**
 * Central API service layer.
 *
 * All backend communication goes through this module.
 * The base URL is read from the VITE_API_BASE_URL environment variable
 * (defaults to http://localhost:8000 for local development).
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

export interface DocumentRecord {
  document_id: string;
  filename: string;
  file_path: string;
  file_size_bytes: number;
  status: string;
  upload_time: string;
  ingestion_triggered: boolean;
  total_chunks: number;
  total_vectors: number;
  error: string | null;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  message: string;
  file_size_bytes: number;
}

export interface DocumentListResponse {
  documents: DocumentRecord[];
  total: number;
}

export interface DocumentDetailResponse {
  document_id: string;
  filename: string;
  file_size_bytes: number;
  status: string;
  upload_time: string;
  ingestion_triggered: boolean;
  total_chunks: number;
  total_vectors: number;
  error: string | null;
  ingestion_status: IngestionStatusResponse | null;
}

export interface StageMetric {
  stage: string;
  duration_seconds: number | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface IngestionStatusResponse {
  document_id: string | null;
  filename: string | null;
  status: string;
  current_stage: string;
  progress: number;
  completed_stages: string[];
  chunks: number;
  text_chunks: number;
  table_chunks: number;
  image_chunks: number;
  vectors: number;
  duration_seconds: number;
  stage_metrics: StageMetric[];
  errors: string[];
}

export interface SearchRequest {
  query: string;
  document_id?: string;
  top_k?: number;
  min_score?: number;
  collection_name?: string;
}

export interface CitationItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  page: number | null;
  content_type: string;
  score: number;
  citation: string;
  content: string;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  query: string;
  answer: string;
  status: string;
  total_results: number;
  results: CitationItem[];
  context: string;
  collection_name: string;
  error: string | null;
}

export interface ComponentHealth {
  name: string;
  status: string;
  message: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  components: ComponentHealth[];
  uptime_seconds: number;
  total_documents: number;
  total_vectors: number;
}

export interface SearchHealthResponse {
  status: string;
  vector_store: string;
  embedding_provider: string;
  collection_exists: boolean;
  collection_name: string;
  message: string;
}

// ── Error handling ────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;
  detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  let response: Response;

  try {
    response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
  } catch (networkErr) {
    throw new ApiError(
      `Cannot reach server at ${BASE_URL}. Is the backend running?`,
      0
    );
  }

  if (!response.ok) {
    let detail: unknown;
    try {
      detail = await response.json();
    } catch {
      detail = await response.text();
    }
    const message =
      typeof detail === "object" && detail !== null && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `Request failed with status ${response.status}`;
    throw new ApiError(message, response.status, detail);
  }

  if (response.status === 204) return undefined as T;

  return response.json() as Promise<T>;
}

// ── Document API ──────────────────────────────────────────────────────────────

export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  // Use XHR for upload progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE_URL}/api/documents/upload`);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          reject(new ApiError("Invalid server response", xhr.status));
        }
      } else {
        let detail: unknown;
        try {
          detail = JSON.parse(xhr.responseText);
        } catch {
          detail = xhr.responseText;
        }
        const msg =
          typeof detail === "object" &&
          detail !== null &&
          "detail" in detail
            ? String((detail as { detail: unknown }).detail)
            : `Upload failed (${xhr.status})`;
        reject(new ApiError(msg, xhr.status, detail));
      }
    };

    xhr.onerror = () =>
      reject(
        new ApiError(
          `Cannot reach server at ${BASE_URL}. Is the backend running?`,
          0
        )
      );
    xhr.ontimeout = () =>
      reject(new ApiError("Upload timed out. Please try again.", 0));

    xhr.timeout = 120_000;
    xhr.send(formData);
  });
}

export async function getDocuments(): Promise<DocumentListResponse> {
  return request<DocumentListResponse>("/api/documents");
}

export async function getDocument(
  documentId: string
): Promise<DocumentDetailResponse> {
  return request<DocumentDetailResponse>(`/api/documents/${documentId}`);
}

export async function deleteDocument(documentId: string): Promise<void> {
  await request<void>(`/api/documents/${documentId}`, { method: "DELETE" });
}

export async function reingestDocument(
  documentId: string
): Promise<DocumentUploadResponse> {
  return request<DocumentUploadResponse>(
    `/api/documents/${documentId}/ingest`,
    { method: "POST" }
  );
}

// ── Ingestion API ─────────────────────────────────────────────────────────────

export async function getIngestionStatus(
  documentId: string
): Promise<IngestionStatusResponse> {
  return request<IngestionStatusResponse>(
    `/api/ingestion/${documentId}/status`
  );
}

// ── Search API ────────────────────────────────────────────────────────────────

export async function searchDocuments(
  req: SearchRequest
): Promise<SearchResponse> {
  return request<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

// ── Health API ────────────────────────────────────────────────────────────────

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function getSearchHealth(): Promise<SearchHealthResponse> {
  return request<SearchHealthResponse>("/api/search/health");
}
