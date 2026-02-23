import axios from 'axios';

// In dev (Codespaces), use the Vite proxy at /api which forwards to FastAPI.
// In production, set VITE_API_URL to the actual backend URL.
const API_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_URL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface PredictionRequest {
  image_base64: string;
}

export interface PredictionResponse {
  prediction: string;
  probability: number;
  confidence: number;
  risk_level: string;
  risk_color: string;
  gradcam_base64: string | null;
  inference_time_ms: number;
  patient_context: {
    morphological_features: string;
  };
}

// ── Multi-cell (auto-segmentation) response ──────────────

export interface CellResult {
  cell_index: number;
  prediction: string;
  probability: number;
  confidence: number;
  risk_level: string;
  risk_color: string;
  gradcam_base64: string | null;
}

export interface MultiCellResponse {
  is_multi_cell: boolean;
  num_cells: number;
  overall_prediction: string;
  overall_risk_level: string;
  overall_risk_color: string;
  blast_count: number;
  normal_count: number;
  blast_percentage: number;
  cells: CellResult[];
  annotated_image_base64: string | null;
  inference_time_ms: number;
  segmentation_message: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device: string;
  timestamp: string;
}

export interface ModelInfoResponse {
  architecture: string;
  backbone: string;
  total_parameters: number;
  trainable_parameters: number;
  num_tabular_features: number;
  feature_names: string[];
  device: string;
  input_size: string;
}

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health');
  return data;
}

export async function getModelInfo(): Promise<ModelInfoResponse> {
  const { data } = await api.get<ModelInfoResponse>('/model/info');
  return data;
}

export async function predict(request: PredictionRequest): Promise<PredictionResponse> {
  const { data } = await api.post<PredictionResponse>('/predict', request);
  return data;
}

export async function predictMultiCell(request: PredictionRequest): Promise<MultiCellResponse> {
  const { data } = await api.post<MultiCellResponse>('/predict/multi', request);
  return data;
}

export async function predictWithUpload(file: File): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await api.post<PredictionResponse>('/predict/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      const result = reader.result as string;
      // Remove the data:image/...;base64, prefix
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
  });
}

// ── History / Records ─────────────────────────────────────

export interface AnalysisRecord {
  id: string;
  prediction: string;
  probability: number;
  confidence: number;
  risk_level: string;
  risk_color: string;
  inference_time_ms: number;
  image_filename: string | null;
  gradcam_base64: string | null;
  created_at: string;
}

export interface AnalysesResponse {
  records: AnalysisRecord[];
  count: number;
  limit: number;
  offset: number;
}

export interface AnalysisStats {
  total_analyses: number;
  aml_detected: number;
  normal_detected: number;
  avg_confidence: number;
  avg_inference_ms: number;
  risk_distribution: Record<string, number>;
}

export async function getAnalyses(limit = 50, offset = 0): Promise<AnalysesResponse> {
  const { data } = await api.get<AnalysesResponse>('/analyses', {
    params: { limit, offset },
  });
  return data;
}

export async function getAnalysisStats(): Promise<AnalysisStats> {
  const { data } = await api.get<AnalysisStats>('/analyses/stats');
  return data;
}

export async function deleteAnalysis(id: string): Promise<void> {
  await api.delete(`/analyses/${id}`);
}

// ── Auth ─────────────────────────────────────────────────────

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  display_name?: string;
  sex: string;
}

export interface LoginData {
  username: string;
  password: string;
}

export interface AuthUser {
  id: string;
  username: string;
  email: string;
  display_name: string;
  sex: string;
  created_at: string;
}

export interface AuthResponse {
  token: string;
  user: AuthUser;
}

export async function registerUser(data: RegisterData): Promise<AuthResponse> {
  const { data: res } = await api.post<AuthResponse>('/auth/register', data);
  return res;
}

export async function loginUser(data: LoginData): Promise<AuthResponse> {
  const { data: res } = await api.post<AuthResponse>('/auth/login', data);
  return res;
}

export async function getMe(token: string): Promise<AuthUser> {
  const { data } = await api.get<AuthUser>('/auth/me', {
    headers: { Authorization: `Bearer ${token}` },
  });
  return data;
}

export async function googleAuth(credential: string): Promise<AuthResponse> {
  const { data } = await api.post<AuthResponse>('/auth/google', { credential });
  return data;
}

// ── Metrics ──────────────────────────────────────────────────

export interface PlatformMetrics {
  accuracy: number;
  auc_roc: number;
  precision: number;
  recall: number;
  f1_score: number;
  inference_ms: number;
  dataset_size: number;
  dataset_patients: number;
  dataset_source: string;
  model_version: string;
  last_trained: string | null;
}

export async function getMetrics(): Promise<PlatformMetrics> {
  const { data } = await api.get<PlatformMetrics>('/metrics');
  return data;
}

export default api;
