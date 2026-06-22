import React, { useState } from 'react'

const overlay = {
  position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)',
  display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200,
}
const modal = {
  background: 'var(--color-surface)', border: '1px solid var(--color-border)',
  borderRadius: 14, padding: '2rem', width: '100%', maxWidth: 380,
  boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
}
const inputStyle = {
  width: '100%', padding: '0.7rem 0.9rem',
  background: 'var(--color-surface-alt)', border: '1px solid var(--color-border)',
  borderRadius: 7, color: 'var(--color-text)', fontSize: '0.95rem',
  fontFamily: 'inherit', outline: 'none', marginBottom: '0.75rem',
}

export default function LoginModal({ onLogin, onClose }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await onLogin(username, password)
    } catch (err) {
      setError(err.response?.data?.error || 'Login failed. Check your credentials.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={overlay} onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div style={modal}>
        <h2 style={{ marginBottom: '0.25rem', fontSize: '1.2rem' }}>Admin Login</h2>
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
          Access classification history and system statistics.
        </p>

        <form onSubmit={handleSubmit}>
          <label style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)' }}>Username</label>
          <input
            style={inputStyle}
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />

          <label style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)' }}>Password</label>
          <input
            style={{ ...inputStyle, marginBottom: error ? '0.5rem' : '1.25rem' }}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />

          {error && (
            <div style={{ color: '#fca5a5', fontSize: '0.85rem', marginBottom: '1rem' }}>
              {error}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button className="btn-primary" type="submit" disabled={loading} style={{ flex: 1 }}>
              {loading ? 'Signing in…' : 'Sign In'}
            </button>
            <button
              type="button"
              style={{
                flex: 1, background: 'none', border: '1px solid var(--color-border)',
                borderRadius: 8, color: 'var(--color-text-muted)', fontSize: '0.95rem',
              }}
              onClick={onClose}
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
