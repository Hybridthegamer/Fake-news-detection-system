import React, { useState } from 'react'
import Header from './components/Header'
import ArticleInput from './components/ArticleInput'
import ResultPanel from './components/ResultPanel'
import AdminDashboard from './components/AdminDashboard'
import LoginModal from './components/LoginModal'
import { classify, login } from './services/api'

const pageStyle = {
  flex: 1,
  maxWidth: 860,
  margin: '0 auto',
  width: '100%',
  padding: '2.5rem 1.5rem',
  display: 'flex',
  flexDirection: 'column',
  gap: '2rem',
}

const heroStyle = {
  textAlign: 'center',
  padding: '2rem 1rem 0.5rem',
}

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showLogin, setShowLogin] = useState(false)
  const [isAdmin, setIsAdmin] = useState(!!localStorage.getItem('admin_token'))

  const handleClassify = async (text) => {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await classify(text)
      setResult(data)
    } catch (err) {
      const msg = err.response?.data?.error || 'Classification failed. Please try again.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleLogin = async (username, password) => {
    const data = await login(username, password)
    localStorage.setItem('admin_token', data.access_token)
    setIsAdmin(true)
    setShowLogin(false)
  }

  const handleLogout = () => {
    localStorage.removeItem('admin_token')
    setIsAdmin(false)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <Header
        onAdminClick={() => setShowLogin(true)}
        isAdmin={isAdmin}
        onLogout={handleLogout}
      />

      <main style={pageStyle}>
        {!isAdmin && (
          <div style={heroStyle}>
            <h1 style={{ fontSize: '2rem', fontWeight: 800, marginBottom: '0.5rem' }}>
              Fake News Detector
            </h1>
            <p style={{ color: 'var(--color-text-muted)', maxWidth: 540, margin: '0 auto', fontSize: '1rem' }}>
              Powered by DistilBERT and TF-IDF/Logistic Regression. Paste any news article or
              headline to receive an AI-powered real/fake classification with confidence score
              and explainability.
            </p>
          </div>
        )}

        {isAdmin ? (
          <AdminDashboard />
        ) : (
          <>
            <div className="card">
              <ArticleInput onSubmit={handleClassify} loading={loading} error={error} />
            </div>

            {loading && (
              <div style={{ textAlign: 'center', color: 'var(--color-text-muted)', padding: '1rem' }}>
                <span style={{ display: 'inline-block', animation: 'spin 1s linear infinite', marginRight: 8 }}>⊙</span>
                Analysing article…
              </div>
            )}

            {result && !loading && <ResultPanel result={result} />}
          </>
        )}
      </main>

      <footer style={{
        textAlign: 'center',
        padding: '1.25rem',
        color: 'var(--color-text-muted)',
        fontSize: '0.8rem',
        borderTop: '1px solid var(--color-border)',
      }}>
        Fake News Detection System — NLP Final Year Project &nbsp;|&nbsp; DistilBERT + TF-IDF/LR
      </footer>

      {showLogin && (
        <LoginModal onLogin={handleLogin} onClose={() => setShowLogin(false)} />
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
      `}</style>
    </div>
  )
}
