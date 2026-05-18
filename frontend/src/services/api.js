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

export const getClaims = (skip = 0, limit = 100) =>
  api.get('/claims/', { params: { skip, limit } });

export const getClaim = (id) => api.get(`/claims/${id}`);

export default api;
