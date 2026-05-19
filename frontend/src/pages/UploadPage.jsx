import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { uploadEdiFile, getDatasetStats, trainModel } from '../services/api';

function UploadCard({ title, description, accent }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState(new Set());
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
    setProgress(0);
    try {
      const res = await uploadEdiFile(file, (e) => {
        if (e.total) {
          setProgress(Math.round((e.loaded / e.total) * 100));
        }
      });
      setProgress(100);
      setResult(res.data);
      setUploadedFiles((prev) => new Set(prev).add(fileName));
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';
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

          {result.errors?.length > 0 && (
            <div className="mt-3">
              <h4 className="font-medium text-red-700 mb-1">Errors</h4>
              <ul className="list-disc list-inside text-red-600 text-xs space-y-0.5">
                {result.errors.map((err, i) => (
                  <li key={i}>{err}</li>
                ))}
              </ul>
            </div>
          )}

          {result.success && (
            <Link
              to="/claims"
              className="inline-block mt-3 text-blue-600 hover:underline font-medium"
            >
              View Claims &rarr;
            </Link>
          )}
        </div>
      )}
    </div>
  );
}

function TrainModelCard() {
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

export default function UploadPage() {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Upload EDI Files</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <UploadCard
          title="837 — Claim Submission"
          description="Upload an 837 file to import claims, service lines, and diagnoses."
          accent="blue"
        />
        <UploadCard
          title="835 — Remittance / Payment"
          description="Upload an 835 file to import payment data, adjustments, and remark codes."
          accent="emerald"
        />
      </div>

      <h1 className="text-2xl font-bold text-gray-900 mt-10 mb-6">Model Training</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <TrainModelCard />
      </div>
    </div>
  );
}
