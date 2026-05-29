function fmtPct(n) {
  if (n == null || Number.isNaN(n)) return '-';
  return `${(n * 100).toFixed(1)}%`;
}

function toneForRate(rate) {
  // Heuristic only — calibrate to your tolerance for new-data inflow.
  if (rate == null) return 'neutral';
  if (rate >= 0.10) return 'bad';
  if (rate >= 0.03) return 'warn';
  return 'good';
}

const TONE_BORDER = {
  neutral: 'border-gray-200',
  good: 'border-green-300',
  warn: 'border-yellow-300',
  bad: 'border-red-300',
};

const TONE_TEXT = {
  neutral: 'text-gray-900',
  good: 'text-green-700',
  warn: 'text-yellow-700',
  bad: 'text-red-700',
};

const DIM_LABEL = {
  payer: 'Payer',
  cpt: 'Procedure',
  dx: 'Diagnosis',
};

function DimensionBlock({ dim, info }) {
  if (!info) return null;
  const top = info.top_unseen_values || [];
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <div className="flex items-baseline justify-between">
        <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
          {DIM_LABEL[dim] || dim}
        </div>
        <div className="text-xs text-gray-400">{info.source_column}</div>
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <div className="text-2xl font-semibold text-gray-900">
          {fmtPct(info.unseen_rate)}
        </div>
        <div className="text-xs text-gray-500">
          {info.unseen_count?.toLocaleString?.() ?? info.unseen_count} unseen
        </div>
      </div>
      {top.length === 0 ? (
        <div className="mt-2 text-xs text-gray-400 italic">No unseen values in window.</div>
      ) : (
        <ul className="mt-2 space-y-1 max-h-40 overflow-y-auto">
          {top.map((row) => (
            <li
              key={row.value}
              className="flex items-center justify-between text-xs"
              title={row.value}
            >
              <span className="text-gray-700 truncate mr-2">{row.value}</span>
              <span className="text-gray-500 tabular-nums">{row.count}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function UnseenRateCard({ data, loading, error, onRefresh, days }) {
  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 text-sm text-gray-500">
        Loading unseen-data rate...
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

  const overallTone = toneForRate(data.any_unseen_rate);

  return (
    <section className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">
            New / Unseen Data
          </h2>
          <p className="text-xs text-gray-500">
            Predictions in the last {days} day{days === 1 ? '' : 's'} whose
            payer, procedure, or diagnosis was not in the model's training
            vocabulary. A rising rate means new data is arriving the model has
            no historical baseline for — treat those risk scores as
            low-confidence.
          </p>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="text-xs text-blue-600 hover:text-blue-800"
          >
            Refresh
          </button>
        )}
      </div>

      <div
        className={`bg-white border ${TONE_BORDER[overallTone]} rounded-lg p-4 shadow-sm`}
      >
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
              Any-Dimension Unseen Rate
            </div>
            <div className={`mt-1 text-3xl font-semibold ${TONE_TEXT[overallTone]}`}>
              {fmtPct(data.any_unseen_rate)}
            </div>
            <div className="mt-1 text-xs text-gray-500">
              {data.any_unseen?.toLocaleString?.() ?? data.any_unseen} of{' '}
              {data.total_predictions?.toLocaleString?.() ?? data.total_predictions}{' '}
              predictions involve at least one unseen category.
            </div>
          </div>
          {data.model_version_filter && (
            <div className="text-xs text-gray-500">
              Filter: <span className="font-mono">{data.model_version_filter}</span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {Object.entries(data.dimensions || {}).map(([dim, info]) => (
          <DimensionBlock key={dim} dim={dim} info={info} />
        ))}
      </div>
    </section>
  );
}
