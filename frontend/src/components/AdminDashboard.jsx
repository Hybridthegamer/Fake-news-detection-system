import React, { useEffect, useState } from 'react'
import { getHistory, getStats } from '../services/api'

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '1.5rem' },
  title: { fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.25rem' },
  statsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
    gap: '0.75rem',
  },
  statCard: {
    background: 'var(--color-surface-alt)',
    border: '1px solid var(--color-border)',
    borderRadius: 10,
    padding: '1rem',
    textAlign: 'center',
  },
  statValue: { fontSize: '1.8rem', fontWeight: 800, color: 'var(--color-primary)' },
  statLabel: { fontSize: '0.78rem', color: 'var(--color-text-muted)', marginTop: 2 },
  tableWrap: { overflowX: 'auto' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' },
  th: {
    padding: '0.6rem 0.9rem', textAlign: 'left',
    borderBottom: '1px solid var(--color-border)',
    color: 'var(--color-text-muted)', fontWeight: 600, fontSize: '0.78rem',
    textTransform: 'uppercase', letterSpacing: '0.06em',
  },
  td: {
    padding: '0.65rem 0.9rem',
    borderBottom: '1px solid #1e2233',
    verticalAlign: 'middle',
  },
  pagination: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    marginTop: '0.75rem',
  },
}

function StatCard({ value, label, color }) {
  return (
    <div style={styles.statCard}>
      <div style={{ ...styles.statValue, color: color || 'var(--color-primary)' }}>{value ?? '—'}</div>
      <div style={styles.statLabel}>{label}</div>
    </div>
  )
}

export default function AdminDashboard() {
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getStats(), getHistory(page)])
      .then(([s, h]) => { setStats(s); setHistory(h) })
      .catch(() => setError('Failed to load dashboard data. Check your session.'))
      .finally(() => setLoading(false))
  }, [page])

  if (loading) return <p style={{ color: 'var(--color-text-muted)', textAlign: 'center', padding: '2rem' }}>Loading dashboard…</p>
  if (error) return <p style={{ color: 'var(--color-fake)', textAlign: 'center', padding: '2rem' }}>{error}</p>

  return (
    <div style={styles.wrapper}>
      <div>
        <h2 style={styles.title}>Admin Dashboard</h2>
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.88rem' }}>
          Classification history and system statistics
        </p>
      </div>

      {stats && (
        <div style={styles.statsGrid}>
          <StatCard value={stats.total_submissions} label="Total Submissions" />
          <StatCard value={stats.real_count} label="Real News" color="var(--color-real)" />
          <StatCard value={stats.fake_count} label="Fake News" color="var(--color-fake)" />
          <StatCard
            value={stats.average_confidence_pct != null ? `${stats.average_confidence_pct}%` : null}
            label="Avg Confidence"
          />
          <StatCard value={stats.distilbert_predictions} label="DistilBERT" />
          <StatCard value={stats.lr_tfidf_predictions} label="LR-TFIDF" />
        </div>
      )}

      {history && (
        <div className="card" style={{ padding: '1rem' }}>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem' }}>
            Recent Submissions
          </h3>
          <div style={styles.tableWrap}>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>#</th>
                  <th style={styles.th}>Article Preview</th>
                  <th style={styles.th}>Type</th>
                  <th style={styles.th}>Verdict</th>
                  <th style={styles.th}>Confidence</th>
                  <th style={styles.th}>Model</th>
                  <th style={styles.th}>Date</th>
                </tr>
              </thead>
              <tbody>
                {history.records.map((r) => {
                  const pred = r.prediction
                  const isFake = pred?.predicted_label === 'Fake'
                  return (
                    <tr key={r.submission_id} style={{ background: 'transparent' }}>
                      <td style={styles.td}>{r.submission_id}</td>
                      <td style={{ ...styles.td, maxWidth: 240, overflow: 'hidden', whiteSpace: 'nowrap', textOverflow: 'ellipsis' }}>
                        {r.article_text}
                      </td>
                      <td style={styles.td}>
                        <span className={`badge ${r.input_type === 'headline' ? 'badge-warn' : ''}`}
                          style={r.input_type !== 'headline' ? { fontSize: '0.78rem' } : {}}>
                          {r.input_type}
                        </span>
                      </td>
                      <td style={styles.td}>
                        {pred ? (
                          <span className={`badge ${isFake ? 'badge-fake' : 'badge-real'}`}>
                            {pred.predicted_label}
                          </span>
                        ) : '—'}
                      </td>
                      <td style={styles.td}>
                        {pred ? `${(pred.confidence_score * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td style={styles.td}>{pred?.model_used || '—'}</td>
                      <td style={{ ...styles.td, color: 'var(--color-text-muted)', fontSize: '0.82rem' }}>
                        {r.submitted_at ? new Date(r.submitted_at).toLocaleString() : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div style={styles.pagination}>
            <span style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)' }}>
              Page {history.current_page} of {history.pages} ({history.total} total)
            </span>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                style={{ background: 'none', border: '1px solid var(--color-border)', borderRadius: 6, padding: '0.3rem 0.7rem', color: 'var(--color-text)', fontSize: '0.85rem' }}
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >
                ← Prev
              </button>
              <button
                style={{ background: 'none', border: '1px solid var(--color-border)', borderRadius: 6, padding: '0.3rem 0.7rem', color: 'var(--color-text)', fontSize: '0.85rem' }}
                disabled={page === history.pages}
                onClick={() => setPage(p => p + 1)}
              >
                Next →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
