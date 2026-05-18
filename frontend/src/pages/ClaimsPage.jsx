import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getClaims } from '../services/api';

const STATUS_STYLES = {
  paid: 'bg-green-100 text-green-800',
  denied: 'bg-red-100 text-red-800',
  submitted: 'bg-yellow-100 text-yellow-800',
  partially_paid: 'bg-orange-100 text-orange-800',
  void: 'bg-gray-100 text-gray-600',
};

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

export default function ClaimsPage() {
  const [claims, setClaims] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    getClaims()
      .then((res) => setClaims(res.data))
      .catch((err) =>
        setError(err.response?.data?.detail || err.message || 'Failed to load claims'),
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <p className="text-gray-500">Loading claims...</p>;
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
        {error}
      </div>
    );
  }

  if (claims.length === 0) {
    return (
      <div className="text-center py-16 text-gray-500">
        <p>No claims found.</p>
        <p className="text-sm mt-1">Upload an EDI file to get started.</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Claims</h1>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              {['Claim #', 'Payer', 'Patient ID', 'Charge', 'Status', 'Service Date', 'Created'].map(
                (h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-left font-medium text-gray-500 uppercase tracking-wider text-xs"
                  >
                    {h}
                  </th>
                ),
              )}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {claims.map((c) => (
              <tr
                key={c.id}
                onClick={() => navigate(`/claims/${c.id}`)}
                className="hover:bg-gray-50 cursor-pointer"
              >
                <td className="px-4 py-3 font-mono whitespace-nowrap">
                  {c.claim_number}
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  {c.payer_name || '-'}
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  {c.patient_member_id || '-'}
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  {fmt(c.total_charge_amount)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <StatusBadge status={c.claim_status} />
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  {c.service_from_date}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-gray-500">
                  {new Date(c.created_at).toLocaleDateString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
