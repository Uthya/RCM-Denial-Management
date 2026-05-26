const SEVERITY_STYLES = {
  stable: 'bg-green-100 text-green-800 border-green-200',
  moderate: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  significant: 'bg-red-100 text-red-800 border-red-200',
};

const CATEGORY_META = {
  structural: {
    title: 'Structural Drift',
    subtitle: 'Missing critical fields · malformed claims · EDI abnormalities',
  },
  business: {
    title: 'Business / Data Distribution Drift',
    subtitle: 'Payer · CPT · Diagnosis · POS · charge distributions',
  },
  operational: {
    title: 'Operational / Parser Drift',
    subtitle: 'Catastrophic null spikes · unexplained score drift · ingestion issues',
  },
};

function SeverityBadge({ severity }) {
  const key = (severity || 'stable').toLowerCase();
  const style = SEVERITY_STYLES[key] || SEVERITY_STYLES.stable;
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-semibold capitalize border ${style}`}
    >
      {key}
    </span>
  );
}

function fmtNum(n, digits = 4) {
  if (n == null || isNaN(n)) return '-';
  return Number(n).toFixed(digits);
}

function fmtPct(n, digits = 2) {
  if (n == null || isNaN(n)) return '-';
  return `${(Number(n) * 100).toFixed(digits)}%`;
}

function CategoryCard({ kind, data, children }) {
  const meta = CATEGORY_META[kind];
  if (!meta) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-900">{meta.title}</h3>
            <SeverityBadge severity={data?.severity} />
          </div>
          <p className="text-xs text-gray-500 mt-0.5">{meta.subtitle}</p>
          {data?.notes && (
            <p className="text-xs text-gray-600 mt-1">{data.notes}</p>
          )}
        </div>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function PsiTable({ categorical = {}, continuous = {} }) {
  const catRows = Object.entries(categorical);
  const contRows = Object.entries(continuous);
  if (catRows.length === 0 && contRows.length === 0) {
    return <p className="text-sm text-gray-500">No distribution comparisons available.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
              Feature
            </th>
            <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
              Kind
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              PSI
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              Unseen / Mean Shift
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              n
            </th>
            <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
              Severity
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {catRows.map(([name, d]) => (
            <tr key={`cat-${name}`} className="hover:bg-gray-50">
              <td className="px-3 py-2 font-mono text-xs text-gray-700">{name}</td>
              <td className="px-3 py-2 text-xs text-gray-500 uppercase">categorical</td>
              <td className="px-3 py-2 text-right tabular-nums">{fmtNum(d.psi)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                {d.unseen_rate != null ? fmtPct(d.unseen_rate) : '-'}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                {d.n ?? '-'}
              </td>
              <td className="px-3 py-2">
                <SeverityBadge severity={d.severity} />
              </td>
            </tr>
          ))}
          {contRows.map(([name, d]) => (
            <tr key={`cont-${name}`} className="hover:bg-gray-50">
              <td className="px-3 py-2 font-mono text-xs text-gray-700">{name}</td>
              <td className="px-3 py-2 text-xs text-gray-500 uppercase">continuous</td>
              <td className="px-3 py-2 text-right tabular-nums">{fmtNum(d.psi)}</td>
              <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                {d.mean_shift_pct != null ? `${d.mean_shift_pct.toFixed(2)}%` : '-'}
              </td>
              <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                {d.n ?? '-'}
              </td>
              <td className="px-3 py-2">
                <SeverityBadge severity={d.severity} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MissingnessTable({ rows = {}, emptyMessage }) {
  const entries = Object.entries(rows);
  if (entries.length === 0) {
    return (
      <p className="text-sm text-gray-500">
        {emptyMessage || 'No missingness deltas in this category.'}
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 text-sm">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
              Column
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              Training
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              Current
            </th>
            <th className="px-3 py-2 text-right font-medium text-gray-500 uppercase tracking-wider text-xs">
              Δ (pp)
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {entries.map(([name, d]) => {
            const delta = d.delta_pct_points ?? 0;
            const tone =
              Math.abs(delta) < 5
                ? 'text-gray-700'
                : Math.abs(delta) < 15
                  ? 'text-yellow-700'
                  : 'text-red-700';
            return (
              <tr key={name} className="hover:bg-gray-50">
                <td className="px-3 py-2 font-mono text-xs text-gray-700">{name}</td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                  {fmtPct(d.reference_rate)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-gray-500">
                  {fmtPct(d.current_rate)}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums font-medium ${tone}`}>
                  {delta > 0 ? '+' : ''}
                  {delta.toFixed(2)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function DriftSummary({ data, loading, error, onRefresh, days }) {
  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-6 text-sm text-gray-500">
        Loading drift report...
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

  const {
    overall = {},
    structural = {},
    business = {},
    operational = {},
    score = {},
    n_current = 0,
    snapshot_trained_at,
  } = data;

  return (
    <section>
      <div className="flex items-center justify-between mb-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Drift Monitoring</h2>
          <p className="text-xs text-gray-500">
            Rolling {days}-day window · {n_current.toLocaleString()} claims compared
            {snapshot_trained_at
              ? ` · snapshot ${new Date(snapshot_trained_at).toLocaleDateString()}`
              : ''}
          </p>
        </div>
        <button
          onClick={onRefresh}
          className="px-3 py-1.5 text-xs font-medium text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
        >
          Refresh
        </button>
      </div>

      {/* Overall + per-category summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
            Overall (worst category)
          </div>
          <div className="mt-2">
            <SeverityBadge severity={overall.severity} />
          </div>
          <div className="mt-1 text-xs text-gray-500">
            {overall.worst_category
              ? `Worst: ${overall.worst_category}`
              : 'All categories stable'}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
            Structural
          </div>
          <div className="mt-2">
            <SeverityBadge severity={structural.severity} />
          </div>
          <div className="mt-1 text-xs text-gray-500">
            Max Δ: {structural.max_delta_pct_points ?? 0} pp
            {structural.max_delta_column ? ` on ${structural.max_delta_column}` : ''}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
            Business
          </div>
          <div className="mt-2">
            <SeverityBadge severity={business.severity} />
          </div>
          <div className="mt-1 text-xs text-gray-500">
            Max PSI: {fmtNum(business.max_psi)}
            {business.max_psi_feature ? ` on ${business.max_psi_feature}` : ''}
          </div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-gray-500 font-medium">
            Operational
          </div>
          <div className="mt-2">
            <SeverityBadge severity={operational.severity} />
          </div>
          <div className="mt-1 text-xs text-gray-500">
            {operational.catastrophic_count ?? 0} catastrophic spike(s) · score PSI{' '}
            {fmtNum(score.psi)}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {/* Structural */}
        <CategoryCard kind="structural" data={structural}>
          <MissingnessTable
            rows={structural.missingness}
            emptyMessage="No missingness data — model snapshot may not include missingness baseline."
          />
        </CategoryCard>

        {/* Business */}
        <CategoryCard kind="business" data={business}>
          <PsiTable
            categorical={business.categorical}
            continuous={business.continuous}
          />
        </CategoryCard>

        {/* Operational */}
        <CategoryCard kind="operational" data={operational}>
          <div className="space-y-4">
            <div>
              <h4 className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-2">
                Catastrophic Null Spikes
              </h4>
              <MissingnessTable
                rows={operational.catastrophic_missingness}
                emptyMessage="No catastrophic missingness spikes detected."
              />
            </div>
            <div>
              <h4 className="text-xs uppercase tracking-wider text-gray-500 font-medium mb-2">
                Score Distribution Drift
              </h4>
              <div className="bg-gray-50 border border-gray-200 rounded-md p-3 text-sm">
                <div className="flex items-center gap-4 flex-wrap">
                  <span>
                    <span className="text-xs text-gray-500">PSI:</span>{' '}
                    <span className="font-mono">{fmtNum(score.psi)}</span>
                  </span>
                  <span>
                    <span className="text-xs text-gray-500">n:</span>{' '}
                    <span className="font-mono">{score.n ?? 0}</span>
                  </span>
                  {score.mean != null && (
                    <span>
                      <span className="text-xs text-gray-500">mean:</span>{' '}
                      <span className="font-mono">{fmtNum(score.mean, 3)}</span>
                    </span>
                  )}
                  {score.p50 != null && (
                    <span>
                      <span className="text-xs text-gray-500">p50:</span>{' '}
                      <span className="font-mono">{fmtNum(score.p50, 3)}</span>
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </CategoryCard>
      </div>
    </section>
  );
}
