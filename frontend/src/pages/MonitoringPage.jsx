import { useCallback, useEffect, useState } from 'react';
import {
  getLivePerformance,
  getDriftReport,
  getPredictionLog,
} from '../services/api';
import LivePerformanceCard from '../components/monitoring/LivePerformanceCard';
import DriftSummary from '../components/monitoring/DriftSummary';
import PredictionLogTable from '../components/monitoring/PredictionLogTable';
import PredictionDetailModal from '../components/monitoring/PredictionDetailModal';

const PAGE_SIZE = 25;

function errorMessage(err) {
  return err?.response?.data?.detail || err?.message || 'Request failed';
}

export default function MonitoringPage() {
  // --- Live performance ---
  const [perfDays, setPerfDays] = useState(30);
  const [perfData, setPerfData] = useState(null);
  const [perfLoading, setPerfLoading] = useState(false);
  const [perfError, setPerfError] = useState(null);

  const loadPerf = useCallback(() => {
    setPerfLoading(true);
    setPerfError(null);
    getLivePerformance({ days: perfDays })
      .then((res) => setPerfData(res.data))
      .catch((err) => setPerfError(errorMessage(err)))
      .finally(() => setPerfLoading(false));
  }, [perfDays]);

  useEffect(() => {
    loadPerf();
  }, [loadPerf]);

  // --- Drift ---
  const [driftDays, setDriftDays] = useState(7);
  const [driftData, setDriftData] = useState(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState(null);

  const loadDrift = useCallback(() => {
    setDriftLoading(true);
    setDriftError(null);
    getDriftReport({ days: driftDays })
      .then((res) => setDriftData(res.data))
      .catch((err) => {
        setDriftData(null);
        setDriftError(errorMessage(err));
      })
      .finally(() => setDriftLoading(false));
  }, [driftDays]);

  useEffect(() => {
    loadDrift();
  }, [loadDrift]);

  // --- Prediction log ---
  const [logRows, setLogRows] = useState([]);
  const [logTotal, setLogTotal] = useState(0);
  const [logOffset, setLogOffset] = useState(0);
  const [logLoading, setLogLoading] = useState(false);
  const [logError, setLogError] = useState(null);
  const [resolvedFilter, setResolvedFilter] = useState('all'); // all | true | false
  const [riskFilter, setRiskFilter] = useState('all');

  const loadLog = useCallback(() => {
    setLogLoading(true);
    setLogError(null);
    getPredictionLog({
      limit: PAGE_SIZE,
      offset: logOffset,
      resolved:
        resolvedFilter === 'all' ? undefined : resolvedFilter === 'true',
      risk_level: riskFilter === 'all' ? undefined : riskFilter,
    })
      .then((res) => {
        setLogRows(res.data.rows || []);
        setLogTotal(res.data.total || 0);
      })
      .catch((err) => setLogError(errorMessage(err)))
      .finally(() => setLogLoading(false));
  }, [logOffset, resolvedFilter, riskFilter]);

  useEffect(() => {
    loadLog();
  }, [loadLog]);

  // --- Modal ---
  const [selectedPrediction, setSelectedPrediction] = useState(null);

  function handleFilterChange(next) {
    if (next.resolved !== undefined) setResolvedFilter(next.resolved);
    if (next.risk_level !== undefined) setRiskFilter(next.risk_level);
    setLogOffset(0);
  }

  function handlePageChange(newPage) {
    setLogOffset(newPage * PAGE_SIZE);
  }

  function refreshAll() {
    loadPerf();
    loadDrift();
    loadLog();
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Monitoring</h1>
          <p className="text-sm text-gray-600 mt-1">
            Live model performance, data drift, and prediction history.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <label className="text-xs text-gray-600 font-medium">Performance window:</label>
            <select
              value={perfDays}
              onChange={(e) => setPerfDays(Number(e.target.value))}
              className="border border-gray-300 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={7}>7 days</option>
              <option value={14}>14 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <label className="text-xs text-gray-600 font-medium">Drift window:</label>
            <select
              value={driftDays}
              onChange={(e) => setDriftDays(Number(e.target.value))}
              className="border border-gray-300 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value={1}>1 day</option>
              <option value={7}>7 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
          <button
            onClick={refreshAll}
            className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
          >
            Refresh All
          </button>
        </div>
      </div>

      <LivePerformanceCard
        data={perfData}
        loading={perfLoading}
        error={perfError}
        onRefresh={loadPerf}
        days={perfDays}
      />

      <DriftSummary
        data={driftData}
        loading={driftLoading}
        error={driftError}
        onRefresh={loadDrift}
        days={driftDays}
      />

      <PredictionLogTable
        rows={logRows}
        total={logTotal}
        offset={logOffset}
        loading={logLoading}
        error={logError}
        resolvedFilter={resolvedFilter}
        riskFilter={riskFilter}
        onFilterChange={handleFilterChange}
        onPageChange={handlePageChange}
        onRowClick={setSelectedPrediction}
        onRefresh={loadLog}
      />

      <PredictionDetailModal
        predictionId={selectedPrediction}
        onClose={() => setSelectedPrediction(null)}
      />
    </div>
  );
}
