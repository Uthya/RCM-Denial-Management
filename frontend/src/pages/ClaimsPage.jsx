import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getClaims } from '../services/api';

const PAGE_SIZE = 100;

const STATUS_STYLES = {
  paid: 'bg-green-100 text-green-800',
  denied: 'bg-red-100 text-red-800',
  submitted: 'bg-yellow-100 text-yellow-800',
  partially_paid: 'bg-orange-100 text-orange-800',
  void: 'bg-gray-100 text-gray-600',
};

const SORTABLE_COLUMNS = [
  { key: 'claim_number', label: 'Claim #' },
  { key: 'payer_name', label: 'Payer' },
  { key: 'total_charge_amount', label: 'Charge' },
  { key: 'claim_status', label: 'Status' },
  { key: 'service_from_date', label: 'Service Date' },
  { key: 'created_at', label: 'Created' },
];

const STATUS_OPTIONS = ['all', 'paid', 'denied', 'submitted', 'partially_paid', 'void'];

function StatusBadge({ status }) {
  const style = STATUS_STYLES[status] || 'bg-gray-100 text-gray-600';
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize ${style}`}
    >
      {status?.replace('_', ' ')}
    </span>
  );
}

function fmt(val) {
  if (val == null) return '-';
  return Number(val).toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
  });
}

function SortIcon({ column, sortBy, sortDir }) {
  if (sortBy !== column) {
    return <span className="ml-1 text-gray-300">&uarr;&darr;</span>;
  }
  return <span className="ml-1">{sortDir === 'asc' ? '\u25B2' : '\u25BC'}</span>;
}

export default function ClaimsPage() {
  const [claims, setClaims] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [sortBy, setSortBy] = useState('created_at');
  const [sortDir, setSortDir] = useState('desc');
  const [statusFilter, setStatusFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const fetchClaims = useCallback(() => {
    setLoading(true);
    setError(null);
    getClaims({
      skip: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      status: statusFilter === 'all' ? undefined : statusFilter,
      sort_by: sortBy,
      sort_dir: sortDir,
    })
      .then((res) => {
        setClaims(res.data.items);
        setTotal(res.data.total);
      })
      .catch((err) =>
        setError(err.response?.data?.detail || err.message || 'Failed to load claims'),
      )
      .finally(() => setLoading(false));
  }, [page, sortBy, sortDir, statusFilter]);

  useEffect(() => {
    fetchClaims();
  }, [fetchClaims]);

  const totalPages = Math.ceil(total / PAGE_SIZE);

  function handleSort(column) {
    if (sortBy === column) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(column);
      setSortDir('asc');
    }
    setPage(0);
  }

  function handleStatusChange(e) {
    setStatusFilter(e.target.value);
    setPage(0);
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
        {error}
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold text-gray-900">Claims</h1>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="status-filter" className="text-gray-600 font-medium">
              Status:
            </label>
            <select
              id="status-filter"
              value={statusFilter}
              onChange={handleStatusChange}
              className="border border-gray-300 rounded-md px-2 py-1 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {STATUS_OPTIONS.map((s) => (
                <option key={s} value={s}>
                  {s === 'all' ? 'All Statuses' : s.replace('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </option>
              ))}
            </select>
          </div>
          <span className="text-sm text-gray-500">
            {total.toLocaleString()} claim{total !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {SORTABLE_COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={() => handleSort(col.key)}
                  className="px-4 py-3 text-left font-medium text-gray-500 uppercase tracking-wider text-xs cursor-pointer select-none hover:text-gray-700"
                >
                  {col.label}
                  <SortIcon column={col.key} sortBy={sortBy} sortDir={sortDir} />
                </th>
              ))}
              <th className="px-4 py-3 text-left font-medium text-gray-500 uppercase tracking-wider text-xs">
                Patient ID
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
            ) : claims.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500">
                  No claims found.
                </td>
              </tr>
            ) : (
              claims.map((c) => (
                <tr
                  key={c.id}
                  onClick={() => navigate(`/claims/${c.id}`)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  <td className="px-4 py-3 font-mono whitespace-nowrap">
                    {c.claim_number}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">{c.payer_name || '-'}</td>
                  <td className="px-4 py-3 whitespace-nowrap">{fmt(c.total_charge_amount)}</td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <StatusBadge status={c.claim_status} />
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">{c.service_from_date}</td>
                  <td className="px-4 py-3 whitespace-nowrap text-gray-500">
                    {new Date(c.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    {c.patient_member_id || '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-4">
          <span className="text-sm text-gray-600">
            Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of{' '}
            {total.toLocaleString()}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage(0)}
              disabled={page === 0}
              className="px-3 py-1 text-sm border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              First
            </button>
            <button
              onClick={() => setPage((p) => p - 1)}
              disabled={page === 0}
              className="px-3 py-1 text-sm border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Prev
            </button>
            <span className="text-sm text-gray-700">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 text-sm border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Next
            </button>
            <button
              onClick={() => setPage(totalPages - 1)}
              disabled={page >= totalPages - 1}
              className="px-3 py-1 text-sm border rounded-md disabled:opacity-40 hover:bg-gray-50"
            >
              Last
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
