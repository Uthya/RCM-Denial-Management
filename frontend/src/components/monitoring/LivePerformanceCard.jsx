function MetricCard({ label, value, subtitle, tone = 'neutral' }) {
  const toneMap = {
    neutral: 'border-gray-200',
    good: 'border-green-300',
    warn: 'border-yellow-300',
    bad: 'border-red-300',
  };
  return (
    <div className={`bg-white border ${toneMap[tone]} rounded-lg p-4 shadow-sm`}>
      <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-gray-900">{value}</div>
      {subtitle && (
        <div className="mt-1 text-xs text-gray-500">{subtitle}</div>
      )}
    </div>
  );
}

function formatPct(n) {
  if (n == null || isNaN(n)) return '-';
  return `${(n * 100).toFixed(2)}%`;
}

function toneForMetric(value) {
  if (value == null) return 'neutral';
  if (value >= 0.8) return 'good';
  if (value >= 0.6) return 'warn';
  return 'bad';
}

export default function LivePerformanceCard({ data, loading, error, onRefresh, days }) {
  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 text-sm text-gray-500">
        Loading live performance...
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-md p-4 text-sm text-red-700">
        {error}
      </div>
    );
  }

  if (!data) return null;

  const { metrics = {}, confusion_matrix: cm = {}, resolved_predictions = 0 } = data;

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Live Model Performance</h2>
          <p className="text-xs text-gray-500">
            Rolling {days}-day window · {resolved_predictions.toLocaleString()} resolved predictions
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="px-3 py-1.5 text-xs font-medium text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      {resolved_predictions === 0 ? (
        <div className="bg-white border border-gray-200 rounded-lg p-6 text-sm text-gray-500">
          No resolved predictions yet. Upload an 835 to reconcile pending predictions.
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricCard
              label="Accuracy"
              value={formatPct(metrics.accuracy)}
              tone={toneForMetric(metrics.accuracy)}
            />
            <MetricCard
              label="Precision"
              value={formatPct(metrics.precision)}
              tone={toneForMetric(metrics.precision)}
            />
            <MetricCard
              label="Recall"
              value={formatPct(metrics.recall)}
              tone={toneForMetric(metrics.recall)}
            />
            <MetricCard
              label="F1 Score"
              value={formatPct(metrics.f1)}
              tone={toneForMetric(metrics.f1)}
            />
          </div>

          <div className="mt-4 bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-3">
              Confusion Matrix
            </div>
            <div className="grid grid-cols-2 gap-2 max-w-md">
              <div className="bg-green-50 border border-green-200 rounded p-3 text-center">
                <div className="text-xs text-gray-600">True Positive</div>
                <div className="text-xl font-semibold text-green-700">{cm.tp ?? 0}</div>
                <div className="text-[10px] text-gray-500 mt-1">Predicted denied · Actually denied</div>
              </div>
              <div className="bg-yellow-50 border border-yellow-200 rounded p-3 text-center">
                <div className="text-xs text-gray-600">False Positive</div>
                <div className="text-xl font-semibold text-yellow-700">{cm.fp ?? 0}</div>
                <div className="text-[10px] text-gray-500 mt-1">Predicted denied · Actually paid</div>
              </div>
              <div className="bg-yellow-50 border border-yellow-200 rounded p-3 text-center">
                <div className="text-xs text-gray-600">False Negative</div>
                <div className="text-xl font-semibold text-yellow-700">{cm.fn ?? 0}</div>
                <div className="text-[10px] text-gray-500 mt-1">Predicted paid · Actually denied</div>
              </div>
              <div className="bg-green-50 border border-green-200 rounded p-3 text-center">
                <div className="text-xs text-gray-600">True Negative</div>
                <div className="text-xl font-semibold text-green-700">{cm.tn ?? 0}</div>
                <div className="text-[10px] text-gray-500 mt-1">Predicted paid · Actually paid</div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
