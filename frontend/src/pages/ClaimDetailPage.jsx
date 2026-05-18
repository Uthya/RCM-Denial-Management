import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getClaim } from '../services/api';

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

function Section({ title, children }) {
  return (
    <section className="mt-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-2">{title}</h2>
      {children}
    </section>
  );
}

export default function ClaimDetailPage() {
  const { id } = useParams();
  const [claim, setClaim] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getClaim(id)
      .then((res) => setClaim(res.data))
      .catch((err) =>
        setError(err.response?.data?.detail || err.message || 'Failed to load claim'),
      )
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <p className="text-gray-500">Loading claim...</p>;
  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
        {error}
      </div>
    );
  }
  if (!claim) return null;

  return (
    <div className="max-w-4xl">
      <Link to="/claims" className="text-sm text-blue-600 hover:underline">
        &larr; Back to Claims
      </Link>

      {/* Header */}
      <div className="mt-4 bg-white rounded-lg border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-3">
          <h1 className="text-xl font-bold text-gray-900 font-mono">
            {claim.claim_number}
          </h1>
          <StatusBadge status={claim.claim_status} />
        </div>
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-2 text-sm">
          <div>
            <dt className="text-gray-500">Payer</dt>
            <dd className="font-medium">{claim.payer_name || '-'}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Patient ID</dt>
            <dd className="font-medium">{claim.patient_member_id || '-'}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Total Charge</dt>
            <dd className="font-medium">{fmt(claim.total_charge_amount)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Service From</dt>
            <dd className="font-medium">{claim.service_from_date}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Service To</dt>
            <dd className="font-medium">{claim.service_to_date || '-'}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Facility Type</dt>
            <dd className="font-medium">{claim.facility_type_code || '-'}</dd>
          </div>
          {claim.previous_payer_claim_control_no && (
            <div>
              <dt className="text-gray-500">Previous Payer Claim Control #</dt>
              <dd className="font-medium font-mono">{claim.previous_payer_claim_control_no}</dd>
            </div>
          )}
        </dl>
      </div>

      {/* Service Lines */}
      {claim.claim_lines?.length > 0 && (
        <Section title="Service Lines">
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  {['#', 'Procedure', 'Mod 1', 'Mod 2', 'Billed', 'Units', 'Service Date', 'POS'].map(
                    (h) => (
                      <th
                        key={h}
                        className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"
                      >
                        {h}
                      </th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {claim.claim_lines.map((l) => (
                  <tr key={l.id}>
                    <td className="px-3 py-2">{l.line_number}</td>
                    <td className="px-3 py-2 font-mono">{l.procedure_code}</td>
                    <td className="px-3 py-2">{l.modifier1 || '-'}</td>
                    <td className="px-3 py-2">{l.modifier2 || '-'}</td>
                    <td className="px-3 py-2">{fmt(l.billed_amount)}</td>
                    <td className="px-3 py-2">{l.units}</td>
                    <td className="px-3 py-2">{l.service_date || '-'}</td>
                    <td className="px-3 py-2">{l.place_of_service || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Diagnoses */}
      {claim.diagnoses?.length > 0 && (
        <Section title="Diagnoses">
          <div className="overflow-x-auto rounded-lg border border-gray-200">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  {['Seq', 'Code', 'Type'].map((h) => (
                    <th
                      key={h}
                      className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {claim.diagnoses.map((d) => (
                  <tr key={d.id}>
                    <td className="px-3 py-2">{d.sequence_number}</td>
                    <td className="px-3 py-2 font-mono">{d.diagnosis_code}</td>
                    <td className="px-3 py-2 capitalize">{d.diagnosis_type}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Remittance / Payment */}
      {claim.remittance_claims?.length > 0 && (
        <Section title="Remittance / Payment">
          {claim.remittance_claims.map((rc) => (
            <div
              key={rc.id}
              className="mb-4 bg-white rounded-lg border border-gray-200 p-4"
            >
              <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-2 text-sm mb-3">
                <div>
                  <dt className="text-gray-500">Status Code</dt>
                  <dd className="font-medium">{rc.claim_status_code}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Billed</dt>
                  <dd className="font-medium">{fmt(rc.billed_amount)}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Paid</dt>
                  <dd className="font-medium">{fmt(rc.paid_amount)}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Remittance Date</dt>
                  <dd className="font-medium">{rc.remittance_date}</dd>
                </div>
                <div>
                  <dt className="text-gray-500">Payer Control #</dt>
                  <dd className="font-medium font-mono">
                    {rc.payer_claim_control_number || '-'}
                  </dd>
                </div>
              </dl>

              {/* Adjustments */}
              {rc.adjustments?.length > 0 && (
                <div className="mt-2">
                  <h4 className="text-xs font-semibold text-gray-500 uppercase mb-1">
                    Adjustments
                  </h4>
                  <div className="overflow-x-auto rounded border border-gray-100">
                    <table className="min-w-full divide-y divide-gray-100 text-xs">
                      <thead className="bg-gray-50">
                        <tr>
                          {['Group', 'Reason Code', 'Amount'].map((h) => (
                            <th
                              key={h}
                              className="px-2 py-1 text-left font-medium text-gray-500 uppercase"
                            >
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {rc.adjustments.map((adj) => (
                          <tr key={adj.id}>
                            <td className="px-2 py-1">{adj.adjustment_group_code}</td>
                            <td className="px-2 py-1 font-mono">
                              {adj.adjustment_reason_code}
                            </td>
                            <td className="px-2 py-1">{fmt(adj.adjustment_amount)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Remark Codes */}
              {rc.remark_codes?.length > 0 && (
                <div className="mt-2">
                  <h4 className="text-xs font-semibold text-gray-500 uppercase mb-1">
                    Remark Codes
                  </h4>
                  <div className="flex flex-wrap gap-1">
                    {rc.remark_codes.map((rm) => (
                      <span
                        key={rm.id}
                        className="inline-block px-2 py-0.5 bg-gray-100 text-gray-700 rounded text-xs font-mono"
                      >
                        {rm.remark_code}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}
