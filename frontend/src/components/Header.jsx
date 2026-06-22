import React from 'react'

const styles = {
  header: {
    background: '#13162280',
    backdropFilter: 'blur(12px)',
    borderBottom: '1px solid var(--color-border)',
    padding: '0 2rem',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  inner: {
    maxWidth: 900,
    margin: '0 auto',
    height: 60,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.6rem',
    fontWeight: 700,
    fontSize: '1.15rem',
    color: 'var(--color-text)',
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: '50%',
    background: 'var(--color-primary)',
    boxShadow: '0 0 8px var(--color-primary)',
  },
  nav: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem',
  },
  navBtn: {
    background: 'none',
    border: '1px solid var(--color-border)',
    color: 'var(--color-text-muted)',
    padding: '0.35rem 0.9rem',
    borderRadius: 6,
    fontSize: '0.88rem',
    transition: 'all 0.2s',
  },
}

export default function Header({ onAdminClick, isAdmin, onLogout }) {
  return (
    <header style={styles.header}>
      <div style={styles.inner}>
        <div style={styles.logo}>
          <span style={styles.dot} />
          FakeNews Detector
        </div>
        <nav style={styles.nav}>
          {isAdmin ? (
            <>
              <span style={{ color: 'var(--color-primary)', fontSize: '0.88rem' }}>
                Admin
              </span>
              <button style={styles.navBtn} onClick={onLogout}>
                Logout
              </button>
            </>
          ) : (
            <button style={styles.navBtn} onClick={onAdminClick}>
              Admin Login
            </button>
          )}
        </nav>
      </div>
    </header>
  )
}
