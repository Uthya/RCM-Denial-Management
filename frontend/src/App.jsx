import { Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import UploadPage from './pages/UploadPage';
import ClaimsPage from './pages/ClaimsPage';
import ClaimDetailPage from './pages/ClaimDetailPage';
import MonitoringPage from './pages/MonitoringPage';

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Sidebar />
      <main className="ml-64 p-6">
        <Routes>
          <Route path="/" element={<Navigate to="/claims" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/claims" element={<ClaimsPage />} />
          <Route path="/claims/:id" element={<ClaimDetailPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
