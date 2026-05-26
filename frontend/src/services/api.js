import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const uploadEdiFile = (file, onUploadProgress) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/edi/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  });
};

export const getEdiFiles = () => api.get('/edi/files');

export const getClaims = ({ skip = 0, limit = 100, status, sort_by, sort_dir } = {}) =>
  api.get('/claims/', { params: { skip, limit, status, sort_by, sort_dir } });

export const getClaim = (id) => api.get(`/claims/${id}`);

export const getDatasetStats = () => api.get('/predictions/dataset-stats');

export const trainModel = () => api.post('/predictions/train');

export const predictClaim = (claimData) => api.post('/predictions/predict', claimData);

export const predictFile = (ediFileId) => api.post(`/predictions/predict-file/${ediFileId}`);

export const predictClaimById = (claimId, signal) =>
  api.post(`/predictions/predict-claim/${claimId}`, null, { signal });

export const getTrainingHistory = ({ skip = 0, limit = 20 } = {}) =>
  api.get('/ml/training-history', { params: { skip, limit } });

export const getLatestTraining = () => api.get('/ml/latest-training');

export const getRecommendationsByFile = (ediFileId, validationErrors = []) =>
  api.post(`/recommendations/by-file/${ediFileId}`, {
    validation_errors: validationErrors,
  });

export default api;
