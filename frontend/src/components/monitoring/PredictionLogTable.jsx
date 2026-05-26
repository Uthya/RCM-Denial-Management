const RISK_STYLES = {
  HIGH: 'bg-red-100 text-red-800',
  MEDIUM: 'bg-yellow-100 text-yellow-800',
  LOW: 'bg-green-100 text-green-800',
};

const OUTCOME_STYLES = {
  denied: 'bg-red-100 text-red-800',
  paid: 'bg-green-100 text-green-800',
  pending: 'bg-gray-100 text-gray-600',
};

const PAGE_SIZE = 25;

function RiskBadge({ level }) {
  const style = RISK_STYLES[level] || 'bg-gray-100 text-gray-600';
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${style}`}>
      {level}
    </span>
  );
}

function OutcomeBadge({ resolvedAt, actualStatus }) {
  if (!resolvedAt) {
    return (
      <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${OUTCOME_STYLES.pending}`}>
        Pending
      </span>
    );
  }
  const style = OUTCOME_STYLES[actualStatus] || OUTCOME_STYLES.pending;
  return (
    <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize ${style}`}>
      {actualStatus || 'resolved'}
    </span>
  );
}

function fmtRisk(score) {
  if (score == null) return '-';
  return Number(score).toFixed(3);
}

function fmtDateTime(iso) {
  if (!iso) return '-';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function PredictionLogTable({
  rows,
  total,
  offset,
  loading,
  error,
  resolvedFilter,
  riskFilter,
  onFilterChange,
  onPageChange,
  onRowClick,
  onRefresh,
}) {
  const totalPages = Math.max(1, Math.ceil((total || 0) / PAGE_SIZE));
  const currentPage = Math.floor(offset / PAGE_SIZE);

  return (
    <section>
      <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Prediction Log</h2>
          <p className="text-xs text-gray-500">
            {(total || 0).toLocaleString()} prediction{total === 1 ? '' : 's'} recorded
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <label className="text-xs text-gray-600 font-medium">Resolved:</label>
          <select
            value={resolvedFilter}
            onChange={(e) => onFilterChange({ resolved: e.target.value })}
            className="border border-gray-300 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All</option>
            <option value="true">Resolved</option>
            <option value="false">Pending</option>
          </select>
          <label className="text-xs text-gray-600 font-medium ml-2">Risk:</label>
          <select
            value={riskFilter}
            onChange={(e) => onFilterChange({ risk_level: e.target.value })}
            className="border border-gray-300 rounded-md px-2 py-1 text-xs bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
          <button
            onClick={onRefresh}
            className="px-3 py-1 text-xs font-medium text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50 ml-2"
          >
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-3 bg-red-50 border border-red-200 rounded-md p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="bg-white overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Prediction ID
              </th>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Claim #
              </th>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Risk Level
              </th>
              <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
                Predicted Risk
              </th>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Actual Outcome
              </th>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Prediction Time
              </th>
              <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Model
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  Loading...
                </td>
              </tr>
            ) : !rows || rows.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  No prediction log entries.
                </td>
              </tr>
            ) : (
              rows.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => onRowClick(r.prediction_id)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  <td className="px-3 py-2 font-mono text-xs text-gray-600 whitespace-nowrap">
                    {r.prediction_id?.slice(0, 8)}…
                  </td>
                  <td className="px-3 py-2 font-mono whitespace-nowrap">
                    {r.claim_number || <span className="text-gray-400">—</span>}
                  </td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    <RiskBadge level={r.risk_level} />
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums whitespace-nowrap">
                    {fmtRisk(r.predicted_risk)}
                  </td>
                  <td className="px-3 py-2 whitespace-nowrap">
                    <OutcomeBadge resolvedAt={r.resolved_at} actualStatus={r.actual_status} />
                  </td>
                  <td className="px-3 py-2 whitespace-nowrap text-gray-500 text-xs">
                    {fmtDateTime(r.prediction_time)}
                  </td>
                  <td className="px-3 py-2 whitespace-nowrap text-gray-500 text-xs">
                    {r.model_version}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-3">
          <span className="text-xs text-gray-600">
            Showing {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total.toLocaleString()}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onPageChange(0)}
              disabled={currentPage === 0}
              className="px-2 py-1 text-xs border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              First
            </button>
            <button
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 0}
              className="px-2 py-1 text-xs border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Prev
            </button>
            <span className="text-xs text-gray-600 px-2">
              {currentPage + 1} / {totalPages}
            </span>
            <button
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage >= totalPages - 1}
              className="px-2 py-1 text-xs border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Next
            </button>
            <button
              onClick={() => onPageChange(totalPages - 1)}
              disabled={currentPage >= totalPages - 1}
              className="px-2 py-1 text-xs border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Last
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
