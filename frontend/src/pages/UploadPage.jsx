import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getDatasetStats,
  getRecommendationsByFile,
  getTrainingHistory,
  predictFile,
  trainModel,
  uploadEdiFile,
} from '../services/api';

const SOURCE_META = {
  parser: { label: 'Parser', cls: 'bg-red-50 text-red-700 border-red-200' },
  carc: { label: 'CARC', cls: 'bg-amber-50 text-amber-800 border-amber-200' },
  model: { label: 'ML', cls: 'bg-indigo-50 text-indigo-700 border-indigo-200' },
};

const STATUS_BADGE_STYLES = {
  Denied: 'bg-red-100 text-red-700 border border-red-200',
  'High Risk': 'bg-orange-100 text-orange-700 border border-orange-200',
  Resolved: 'bg-emerald-100 text-emerald-700 border border-emerald-200',
};

function UploadCard({ title, description, accent, onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState(new Set());
  const [predResult, setPredResult] = useState(null);
  const [predLoading, setPredLoading] = useState(false);
  const [predError, setPredError] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (uploadedFiles.has(f.name)) {
      setError(`"${f.name}" has already been uploaded.`);
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    setProgress(0);
  }, [uploadedFiles]);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile],
  );

  const onUpload = async () => {
    if (!file) return;
    const fileName = file.name;
    setLoading(true);
    setError(null);
    setResult(null);
    setPredResult(null);
    setPredError(null);
    setProgress(0);
    try {
      const res = await uploadEdiFile(file, (e) => {
        if (e.total) {
          setProgress(Math.round((e.loaded / e.total) * 100));
        }
      });
      setProgress(100);
      const data = res.data;
      setResult(data);
      setUploadedFiles((prev) => new Set(prev).add(fileName));
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';

      // Auto-predict for 837 files (file-level summary card)
      if (data.file_type === 'edi_837' && data.edi_file_id && data.claims_count > 0) {
        setPredLoading(true);
        try {
          const predRes = await predictFile(data.edi_file_id);
          setPredResult(predRes.data);
        } catch (predErr) {
          const isModelMissing = predErr.response?.status === 503;
          setPredError(
            isModelMissing
              ? 'Prediction unavailable — train the model first'
              : predErr.response?.data?.detail || 'Prediction failed'
          );
        } finally {
          setPredLoading(false);
        }
      }

      // Bubble up to UploadPage so the Recommendations panel can refresh.
      // Send for both 837 and 835 — the panel decides what to show.
      if (data.edi_file_id) {
        onUploadComplete?.({
          edi_file_id: data.edi_file_id,
          file_type: data.file_type,
          validation_errors: data.validation_errors || [],
        });
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const borderAccent = accent === 'blue' ? 'border-blue-200' : 'border-emerald-200';
  const bgAccent = accent === 'blue' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-emerald-600 hover:bg-emerald-700';
  const barColor = accent === 'blue' ? 'bg-blue-500' : 'bg-emerald-500';
  const barTrack = accent === 'blue' ? 'bg-blue-100' : 'bg-emerald-100';

  return (
    <div className={`rounded-lg border ${borderAccent} bg-white p-5`}>
      <h2 className="text-lg font-semibold text-gray-900 mb-1">{title}</h2>
      <p className="text-sm text-gray-500 mb-4">{description}</p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".edi,.txt,.x12"
          className="hidden"
          onChange={(e) => handleFile(e.target.files?.[0] || null)}
        />
        {file ? (
          <p className="text-gray-700 font-medium">{file.name}</p>
        ) : (
          <>
            <p className="text-gray-500">Drag & drop a file here, or click to browse</p>
            <p className="text-xs text-gray-400 mt-1">.edi, .txt, .x12</p>
          </>
        )}
      </div>

      {/* Upload button */}
      <button
        onClick={onUpload}
        disabled={!file || loading}
        className={`mt-4 w-full px-4 py-2 text-white rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed ${bgAccent}`}
      >
        {loading ? 'Uploading...' : 'Upload & Parse'}
      </button>

      {/* Progress bar */}
      {loading && (
        <div className="mt-3">
          <div className={`w-full h-2.5 rounded-full ${barTrack}`}>
            <div
              className={`h-2.5 rounded-full transition-all duration-300 ${barColor}`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-1 text-right">{progress}%</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div
          className={`mt-4 p-4 rounded-md border text-sm ${
            result.success
              ? 'bg-green-50 border-green-200'
              : 'bg-yellow-50 border-yellow-200'
          }`}
        >
          <h3 className="font-semibold text-gray-900 mb-2">
            {result.success ? 'Parse Successful' : 'Parse Completed with Errors'}
          </h3>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-700">
            <dt>File Type</dt>
            <dd className="font-mono">{result.file_type || '-'}</dd>
            <dt>Claims</dt>
            <dd>{result.claims_count}</dd>
            <dt>Service Lines</dt>
            <dd>{result.claim_lines_count}</dd>
            <dt>Diagnoses</dt>
            <dd>{result.diagnoses_count}</dd>
            <dt>Remittance Claims</dt>
            <dd>{result.remittance_claims_count}</dd>
            <dt>Adjustments</dt>
            <dd>{result.adjustments_count}</dd>
            <dt>Remark Codes</dt>
            <dd>{result.remark_codes_count}</dd>
            <dt>Raw Segments</dt>
            <dd>{result.raw_segments_count}</dd>
          </dl>

          {result.success && !predResult && !predLoading && (
            <Link
              to="/claims"
              className="inline-block mt-3 text-blue-600 hover:underline font-medium"
            >
              View Claims &rarr;
            </Link>
          )}
        </div>
      )}

      {/* Prediction loading */}
      {predLoading && (
        <div className="mt-4 flex items-center gap-2 text-sm text-blue-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Running denial predictions...
        </div>
      )}

      {/* Prediction error (non-blocking warning) */}
      {predError && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-md text-yellow-800 text-sm">
          {predError}
        </div>
      )}

      {/* Prediction summary */}
      {predResult && (() => {
        const total = predResult.predicted_claims || 1;
        const high = predResult.risk_summary?.HIGH || 0;
        const med = predResult.risk_summary?.MEDIUM || 0;
        const low = predResult.risk_summary?.LOW || 0;
        const highPct = Math.round((high / total) * 100);
        const medPct = Math.round((med / total) * 100);
        const lowPct = Math.round((low / total) * 100);

        return (
          <div
            className={`mt-4 p-4 rounded-md border text-sm ${
              high > 0 ? 'bg-red-50 border-red-300' : 'bg-green-50 border-green-200'
            }`}
          >
            <h3 className="font-semibold text-gray-900 mb-1">Upload Processed Successfully</h3>
            <p className="text-gray-700">
              <span className="font-semibold">{total}</span> claims analyzed
            </p>
            {high > 0 ? (
              <p className="text-red-700 font-medium mt-1">
                {high} high-risk claims detected — see Recommended Fixes below
              </p>
            ) : (
              <p className="text-green-700 font-medium mt-1">No high-risk claims detected</p>
            )}

            {/* Risk distribution */}
            <div className="mt-3 grid grid-cols-3 gap-2 text-center">
              <div className="rounded-md bg-red-100 border border-red-200 py-2 px-1">
                <p className="text-lg font-bold text-red-700">{high}</p>
                <p className="text-xs text-red-600 font-medium">HIGH ({highPct}%)</p>
              </div>
              <div className="rounded-md bg-yellow-100 border border-yellow-200 py-2 px-1">
                <p className="text-lg font-bold text-yellow-700">{med}</p>
                <p className="text-xs text-yellow-600 font-medium">MEDIUM ({medPct}%)</p>
              </div>
              <div className="rounded-md bg-green-100 border border-green-200 py-2 px-1">
                <p className="text-lg font-bold text-green-700">{low}</p>
                <p className="text-xs text-green-600 font-medium">LOW ({lowPct}%)</p>
              </div>
            </div>

            <Link
              to="/claims"
              className="inline-block mt-3 text-blue-600 hover:underline font-medium"
            >
              View Claims &rarr;
            </Link>
          </div>
        );
      })()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// NEW: Recommended Fixes panel (left side of Row 2)
// Single-expand accordion: only one claim can be open at a time.
// ---------------------------------------------------------------------------

function RecommendedFixesPanel({ data, loading, error, title, subtitle, emptyText }) {
  const [openClaimId, setOpenClaimId] = useState(null);

  // When the dataset is replaced (new upload), collapse any open card.
  useEffect(() => {
    setOpenClaimId(null);
  }, [data?.edi_file_id]);

  const claims = data?.claims || [];

  return (
    <div className="rounded-lg border border-rose-200 bg-white p-5 flex flex-col">
      <h2 className="text-lg font-semibold text-gray-900 mb-1">
        {title || 'Recommended Fixes for High-Risk / Denied Claims'}
      </h2>
      <p className="text-sm text-gray-500 mb-4">
        {subtitle ||
          'Per-claim corrective actions derived from parser findings, payer CARC codes, and ML risk factors.'}
      </p>

      {loading && (
        <div className="flex items-center gap-2 text-sm text-rose-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Generating recommendations…
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {!loading && !error && !data && (
        <div className="text-sm text-gray-400 border border-dashed border-gray-200 rounded-md p-6 text-center">
          {emptyText ||
            'Recommendations will appear here after the next 837 or 835 upload.'}
        </div>
      )}

      {!loading && !error && data && claims.length === 0 && (
        <div className="text-sm text-gray-500 border border-dashed border-gray-200 rounded-md p-6 text-center">
          No flagged claims in this upload. All claims look healthy.
        </div>
      )}

      {!loading && !error && claims.length > 0 && (
        <>
          <div className="text-xs text-gray-500 mb-2">
            {data.flagged_claims} of {data.total_claims_in_file} claims need attention &middot; file #{data.edi_file_id}
          </div>
          <ul className="space-y-1.5 max-h-[480px] overflow-y-auto pr-1">
            {claims.map((c) => {
              const isResolved = c.resolved || c.status_badge === 'Resolved';
              const isOpen = !isResolved && openClaimId === c.claim_id;
              const badgeCls = STATUS_BADGE_STYLES[c.status_badge] || 'bg-gray-100 text-gray-700 border';
              return (
                <li
                  key={c.claim_id}
                  className={`rounded-md border overflow-hidden ${
                    isResolved
                      ? 'border-emerald-200 bg-emerald-50/40'
                      : 'border-gray-200 bg-white'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => !isResolved && setOpenClaimId(isOpen ? null : c.claim_id)}
                    aria-expanded={isOpen}
                    disabled={isResolved}
                    className={`w-full flex items-center justify-between gap-3 px-3 py-2.5 transition-colors ${
                      isResolved ? 'cursor-default' : 'hover:bg-gray-50'
                    }`}
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      {isResolved ? (
                        <span className="text-emerald-600 text-sm leading-none" aria-hidden="true">
                          &#10003;
                        </span>
                      ) : (
                        <span
                          className={`text-gray-400 text-xs transition-transform ${
                            isOpen ? 'rotate-90' : ''
                          }`}
                        >
                          &#9656;
                        </span>
                      )}
                      <span className="font-mono text-sm text-gray-900 font-semibold truncate">
                        {c.claim_number}
                      </span>
                      {c.payer_name && (
                        <span className="text-xs text-gray-400 truncate hidden sm:inline">
                          &middot; {c.payer_name}
                        </span>
                      )}
                    </span>
                    <span className="flex items-center gap-2 flex-shrink-0">
                      {isResolved ? (
                        <span className="text-[11px] text-emerald-700 italic">
                          Fix verified
                        </span>
                      ) : (
                        c.risk_score != null && (
                          <span className="text-[11px] font-mono text-gray-500">
                            {(c.risk_score * 100).toFixed(0)}%
                          </span>
                        )
                      )}
                      <span
                        className={`text-[11px] font-semibold rounded px-2 py-0.5 ${badgeCls}`}
                      >
                        {c.status_badge}
                      </span>
                    </span>
                  </button>
                  {isOpen && (
                    <div className="border-t border-gray-100 bg-gray-50/60 px-3 py-3 space-y-2">
                      {c.recommendations.map((rec, i) => {
                        const sm = SOURCE_META[rec.source] || SOURCE_META.parser;
                        return (
                          <div
                            key={i}
                            className="bg-white rounded-md border border-gray-200 px-3 py-2.5"
                          >
                            <div className="flex items-center justify-between gap-2 mb-1.5">
                              <span
                                className={`text-[10px] font-semibold uppercase tracking-wide rounded px-1.5 py-0.5 border ${sm.cls}`}
                              >
                                {sm.label}
                              </span>
                              {rec.location && (
                                <span className="text-[10px] text-gray-400 font-mono truncate">
                                  {rec.location}
                                </span>
                              )}
                            </div>
                            <div>
                              <p className="text-[11px] uppercase tracking-wide text-gray-500 font-medium">
                                Reason for Denial
                              </p>
                              <p className="text-sm text-gray-900 break-words">
                                {rec.reason}
                              </p>
                            </div>
                            <div className="mt-1.5">
                              <p className="text-[11px] uppercase tracking-wide text-gray-500 font-medium">
                                Recommended Fix
                              </p>
                              <p className="text-sm text-gray-700">{rec.fix}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

function TrainModelCard({ onTrainingComplete }) {
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fetchStats = useCallback(async () => {
    setStatsLoading(true);
    try {
      const res = await getDatasetStats();
      setStats(res.data);
    } catch (err) {
      setStats(null);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const onTrain = async () => {
    setTraining(true);
    setError(null);
    setResult(null);
    try {
      const res = await trainModel();
      setResult(res.data);
      fetchStats();
      onTrainingComplete?.();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="rounded-lg border border-violet-200 bg-white p-5">
      <h2 className="text-lg font-semibold text-gray-900 mb-1">Train Denial Prediction Model</h2>
      <p className="text-sm text-gray-500 mb-4">
        Train the XGBoost model using adjudicated claims from the database.
      </p>

      {/* Dataset stats */}
      <div className="rounded-md bg-gray-50 border border-gray-200 p-4 mb-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Dataset Overview</h3>
        {statsLoading ? (
          <p className="text-sm text-gray-400">Loading stats...</p>
        ) : stats ? (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-gray-700">
            <dt>Matched Claims</dt>
            <dd className="font-semibold">{stats.total} claims</dd>
            <dt>Denied</dt>
            <dd className="font-mono text-red-600">{stats.denied}</dd>
            <dt>Paid</dt>
            <dd className="font-mono text-green-600">{stats.paid}</dd>
            <dt>Denial Rate</dt>
            <dd className="font-mono">{(stats.denial_rate * 100).toFixed(1)}%</dd>
          </dl>
        ) : (
          <p className="text-sm text-gray-400">No labelled claims found</p>
        )}
      </div>

      {/* Train button */}
      <button
        onClick={onTrain}
        disabled={training || !stats || stats.total === 0}
        className="w-full px-4 py-2 text-white rounded-md font-medium disabled:opacity-50 disabled:cursor-not-allowed bg-violet-600 hover:bg-violet-700"
      >
        {training ? 'Training Model...' : 'Train Model'}
      </button>

      {/* Training spinner */}
      {training && (
        <div className="mt-3 flex items-center gap-2 text-sm text-violet-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Training on {stats?.total || 0} matched claims...
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {/* Training result */}
      {result && (
        <div className="mt-4 p-4 rounded-md border bg-green-50 border-green-200 text-sm">
          <h3 className="font-semibold text-gray-900 mb-2">Training Complete</h3>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-gray-700">
            <dt>Status</dt>
            <dd className="font-semibold text-green-700">{result.status}</dd>
            <dt>Training Samples</dt>
            <dd>{result.split?.train_samples}</dd>
            <dt>Test Samples</dt>
            <dd>{result.split?.test_samples}</dd>
            <dt>Accuracy</dt>
            <dd className="font-mono">{(result.metrics?.accuracy * 100).toFixed(1)}%</dd>
            <dt>Precision</dt>
            <dd className="font-mono">{(result.metrics?.precision * 100).toFixed(1)}%</dd>
            <dt>Recall</dt>
            <dd className="font-mono">{(result.metrics?.recall * 100).toFixed(1)}%</dd>
            <dt>F1 Score</dt>
            <dd className="font-mono">{(result.metrics?.f1 * 100).toFixed(1)}%</dd>
            {result.metrics?.roc_auc != null && (
              <>
                <dt>ROC-AUC</dt>
                <dd className="font-mono">{(result.metrics.roc_auc * 100).toFixed(1)}%</dd>
              </>
            )}
            <dt>Training Time</dt>
            <dd className="font-mono">{result.training_time_seconds}s</dd>
          </dl>
        </div>
      )}
    </div>
  );
}

function TrainingHistoryCard({ refreshKey }) {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getTrainingHistory({ skip: 0, limit: 10 });
      setItems(res.data?.items || []);
      setTotal(res.data?.total || 0);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory, refreshKey]);

  const fmtTimestamp = (iso) => {
    if (!iso) return '—';
    try {
      return new Date(iso).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      });
    } catch {
      return iso;
    }
  };
  const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

  return (
    <div className="rounded-lg border border-indigo-200 bg-white p-5">
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg font-semibold text-gray-900">Training History</h2>
        <button
          type="button"
          onClick={fetchHistory}
          disabled={loading}
          className="text-xs text-indigo-600 hover:underline disabled:opacity-50"
        >
          Refresh
        </button>
      </div>
      <p className="text-sm text-gray-500 mb-4">
        Performance metrics from previous training runs.
      </p>

      {error && (
        <div className="p-3 mb-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      {loading && items.length === 0 ? (
        <p className="text-sm text-gray-400">Loading history...</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-gray-400">No training runs recorded yet.</p>
      ) : (
        <>
          <ul className="space-y-2 max-h-[420px] overflow-y-auto pr-1">
            {items.map((run) => (
              <li
                key={run.training_id}
                className="rounded-md border border-gray-200 bg-gray-50 px-3 py-2.5"
              >
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold text-gray-900">
                    {fmtTimestamp(run.training_timestamp)}
                  </p>
                  {run.model_version && (
                    <span className="text-[11px] font-mono text-indigo-700 bg-indigo-50 border border-indigo-100 rounded px-1.5 py-0.5">
                      {run.model_version}
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5">
                  {run.total_claims_used.toLocaleString()} claims ·{' '}
                  {run.training_samples.toLocaleString()} train /{' '}
                  {run.test_samples.toLocaleString()} test
                </p>
                <dl className="grid grid-cols-4 gap-x-2 gap-y-0.5 mt-2 text-xs">
                  <div>
                    <dt className="text-gray-500">Acc</dt>
                    <dd className="font-mono text-gray-900">{pct(run.accuracy)}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-500">F1</dt>
                    <dd className="font-mono text-gray-900">{pct(run.f1_score)}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-500">Prec</dt>
                    <dd className="font-mono text-gray-900">{pct(run.precision)}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-500">Rec</dt>
                    <dd className="font-mono text-gray-900">{pct(run.recall)}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-500">AUC</dt>
                    <dd className="font-mono text-gray-900">{pct(run.roc_auc)}</dd>
                  </div>
                  <div>
                    <dt className="text-gray-500">Denial</dt>
                    <dd className="font-mono text-gray-900">{pct(run.denial_rate)}</dd>
                  </div>
                  <div className="col-span-2">
                    <dt className="text-gray-500">Time</dt>
                    <dd className="font-mono text-gray-900">
                      {run.training_time_seconds?.toFixed(2)}s
                    </dd>
                  </div>
                </dl>
              </li>
            ))}
          </ul>
          {total > items.length && (
            <p className="text-xs text-gray-400 mt-2">
              Showing {items.length} of {total} runs
            </p>
          )}
        </>
      )}
    </div>
  );
}

async function fetchRecAndSet(setter, ediFileId, validationErrors) {
  setter((prev) => ({ data: prev.data, loading: true, error: null }));
  try {
    const res = await getRecommendationsByFile(ediFileId, validationErrors);
    setter({ data: res.data, loading: false, error: null });
  } catch (err) {
    const status = err.response?.status;
    setter({
      data: null,
      loading: false,
      error:
        status === 503
          ? 'Predictions unavailable — train the model first to surface ML-based fixes.'
          : err.response?.data?.detail || err.message || 'Failed to load recommendations',
    });
  }
}

export default function UploadPage() {
  const [trainingVersion, setTrainingVersion] = useState(0);
  // Per-file-type recommendation state so an 837 upload and an 835 upload can
  // coexist side by side instead of overwriting each other.
  const [rec837, setRec837] = useState({ data: null, loading: false, error: null });
  const [rec835, setRec835] = useState({ data: null, loading: false, error: null });

  // Refs so the upload handler can refresh the *other* panel without
  // re-creating the callback on every state change.
  const rec837Ref = useRef(rec837);
  const rec835Ref = useRef(rec835);
  useEffect(() => { rec837Ref.current = rec837; }, [rec837]);
  useEffect(() => { rec835Ref.current = rec835; }, [rec835]);

  const handleUploadComplete = useCallback(
    async ({ edi_file_id, file_type, validation_errors }) => {
      const isReplacement = file_type === 'edi_835';
      const primarySetter = isReplacement ? setRec835 : setRec837;
      const otherSetter = isReplacement ? setRec837 : setRec835;
      const otherCurrent = isReplacement ? rec837Ref.current : rec835Ref.current;

      // Update the panel matching the just-uploaded file type
      const primaryFetch = fetchRecAndSet(primarySetter, edi_file_id, validation_errors);

      // Re-evaluate the other panel against its last-known file id so any
      // claims that just became "Resolved" reflect immediately.
      const otherFetch = otherCurrent.data?.edi_file_id
        ? fetchRecAndSet(otherSetter, otherCurrent.data.edi_file_id, [])
        : Promise.resolve();

      await Promise.all([primaryFetch, otherFetch]);
    },
    []
  );

  // Row 2 visibility: appear only when there's an actionable fix to show
  // (any unresolved claim in either panel), or while a fetch is in flight,
  // or when an error needs to be surfaced. A panel populated entirely with
  // "Resolved" claims is *not* actionable, so the row hides in that case.
  const has837Fix = (rec837.data?.claims || []).some((c) => !c.resolved);
  const has835Fix = (rec835.data?.claims || []).some((c) => !c.resolved);
  const showRecommendationRow =
    rec837.loading || rec835.loading ||
    rec837.error || rec835.error ||
    has837Fix || has835Fix;

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Upload EDI Files</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <UploadCard
          title="837 — Claim Submission"
          description="Upload an 837 file to import claims, service lines, and diagnoses."
          accent="blue"
          onUploadComplete={handleUploadComplete}
        />
        <UploadCard
          title="835 — Remittance / Payment"
          description="Upload an 835 file to import payment data, adjustments, and remark codes."
          accent="emerald"
          onUploadComplete={handleUploadComplete}
        />
      </div>

      {/* Row 2: Recommended Fixes — only rendered when there is an actionable fix */}
      {showRecommendationRow && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10">
          <RecommendedFixesPanel
            data={rec837.data}
            loading={rec837.loading}
            error={rec837.error}
            title="Recommended Fixes — 837 Submission"
            subtitle="High-risk and denied claims from the latest 837 upload, with ML- and parser-derived fixes."
            emptyText="Recommendations will appear here after the next 837 upload."
          />
          <RecommendedFixesPanel
            data={rec835.data}
            loading={rec835.loading}
            error={rec835.error}
            title="Recommended Fixes — 835 Adjudication"
            subtitle="Denied claims from the latest 835 upload, with CARC-derived fixes."
            emptyText="Recommendations will appear here after the next 835 upload."
          />
        </div>
      )}

      <h1 className="text-2xl font-bold text-gray-900 mt-10 mb-6">Model Training</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <TrainModelCard
          onTrainingComplete={() => setTrainingVersion((v) => v + 1)}
        />
        <TrainingHistoryCard refreshKey={trainingVersion} />
      </div>
    </div>
  );
}
