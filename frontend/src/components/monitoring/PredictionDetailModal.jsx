import { useEffect, useState } from 'react';
import { getPredictionLogDetail } from '../../services/api';

function Field({ label, value, mono = false }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
        {label}
      </div>
      <div
        className={`mt-0.5 text-sm text-gray-900 break-all ${mono ? 'font-mono' : ''}`}
      >
        {value ?? <span className="text-gray-400">—</span>}
      </div>
    </div>
  );
}

function fmtDateTime(iso) {
  if (!iso) return null;
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function PredictionDetailModal({ predictionId, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!predictionId) return;
    setLoading(true);
    setError(null);
    setData(null);
    getPredictionLogDetail(predictionId)
      .then((res) => setData(res.data))
      .catch((err) =>
        setError(err.response?.data?.detail || err.message || 'Failed to load prediction')
      )
      .finally(() => setLoading(false));
  }, [predictionId]);

  if (!predictionId) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto"
      >
        <div className="px-5 py-3 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-gray-900">Prediction Detail</h3>
            <p className="text-xs text-gray-500 font-mono mt-0.5">{predictionId}</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-xl leading-none"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <div className="p-5">
          {loading && (
            <div className="text-sm text-gray-500 py-6 text-center">Loading...</div>
          )}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">
              {error}
            </div>
          )}
          {data && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-5">
                <Field label="Claim #" value={data.claim_number} mono />
                <Field label="Claim ID" value={data.claim_id ?? '—'} />
                <Field label="Risk Level" value={data.risk_level} />
                <Field label="Predicted Risk" value={data.predicted_risk?.toFixed(4)} mono />
                <Field label="Predicted Label" value={data.predicted_label === 1 ? 'Denied' : 'Paid'} />
                <Field
                  label="Prediction Time"
                  value={fmtDateTime(data.prediction_time)}
                />
                <Field label="Model Version" value={data.model_version} mono />
                <Field
                  label="Feature Version"
                  value={data.feature_engineering_version}
                  mono
                />
                <Field label="Actual Outcome" value={data.actual_status || 'Pending'} />
                {data.actual_denied != null && (
                  <Field
                    label="Actual Denied"
                    value={data.actual_denied === 1 ? 'Yes' : 'No'}
                  />
                )}
                <Field label="Resolved At" value={fmtDateTime(data.resolved_at)} />
                <Field
                  label="Resolved By Remittance"
                  value={data.resolved_by_remittance_id ?? '—'}
                  mono
                />
              </div>

              {data.actual_denied != null && (
                <div className="mb-5 p-3 rounded-md border bg-gray-50 border-gray-200">
                  <div className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-1">
                    Outcome vs. Prediction
                  </div>
                  <div className="text-sm text-gray-800">
                    {data.predicted_label === data.actual_denied ? (
                      <span className="text-green-700 font-medium">
                        ✓ Correct prediction
                      </span>
                    ) : (
                      <span className="text-red-700 font-medium">
                        ✗ Incorrect — predicted{' '}
                        {data.predicted_label === 1 ? 'denied' : 'paid'}, actually{' '}
                        {data.actual_denied === 1 ? 'denied' : 'paid'}
                      </span>
                    )}
                  </div>
                </div>
              )}

              <div>
                <div className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-2">
                  Feature Snapshot
                </div>
                {data.feature_snapshot ? (
                  <pre className="bg-gray-900 text-gray-100 text-xs rounded-md p-4 overflow-x-auto">
                    {JSON.stringify(data.feature_snapshot, null, 2)}
                  </pre>
                ) : (
                  <div className="text-sm text-gray-500">No snapshot recorded.</div>
                )}
              </div>
            </>
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm font-medium text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
