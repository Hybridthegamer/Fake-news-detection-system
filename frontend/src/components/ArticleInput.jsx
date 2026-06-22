import React, { useState } from 'react'

const PLACEHOLDER = `Paste a news article or headline here…

Example: "Scientists at [University] have discovered a new treatment for cancer that cures patients overnight with zero side effects, but the government is suppressing it."`

const styles = {
  wrapper: { display: 'flex', flexDirection: 'column', gap: '1rem' },
  label: { fontSize: '0.9rem', color: 'var(--color-text-muted)', fontWeight: 500 },
  textarea: {
    width: '100%',
    minHeight: 200,
    background: 'var(--color-surface-alt)',
    border: '1px solid var(--color-border)',
    borderRadius: 8,
    color: 'var(--color-text)',
    fontSize: '0.95rem',
    lineHeight: 1.7,
    padding: '1rem',
    resize: 'vertical',
    outline: 'none',
    fontFamily: 'inherit',
    transition: 'border-color 0.2s',
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: '0.75rem',
  },
  wordCount: { fontSize: '0.82rem', color: 'var(--color-text-muted)' },
  hint: { fontSize: '0.82rem', color: 'var(--color-text-muted)' },
  error: {
    background: 'var(--color-fake-bg)',
    border: '1px solid var(--color-fake)',
    borderRadius: 6,
    padding: '0.6rem 0.9rem',
    color: '#fca5a5',
    fontSize: '0.88rem',
  },
}

function countWords(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0
}

export default function ArticleInput({ onSubmit, loading, error }) {
  const [text, setText] = useState('')
  const wc = countWords(text)
  const tooShort = wc > 0 && wc < 5
  const tooLong = wc > 5000

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!text.trim() || tooShort || tooLong) return
    onSubmit(text)
  }

  return (
    <form style={styles.wrapper} onSubmit={handleSubmit}>
      <label style={styles.label} htmlFor="article-input">
        News Article or Headline
      </label>

      <textarea
        id="article-input"
        style={{
          ...styles.textarea,
          borderColor: tooShort || tooLong ? 'var(--color-fake)' : 'var(--color-border)',
        }}
        placeholder={PLACEHOLDER}
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={loading}
        aria-label="News article text input"
      />

      {error && <div style={styles.error}>{error}</div>}

      <div style={styles.footer}>
        <span style={styles.wordCount}>
          {wc > 0 ? (
            <span style={{ color: tooShort || tooLong ? 'var(--color-fake)' : 'inherit' }}>
              {wc.toLocaleString()} words
              {tooShort && ' — minimum 5 words'}
              {tooLong && ' — maximum 5,000 words'}
            </span>
          ) : (
            'Word count: 0'
          )}
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          {text && (
            <button
              type="button"
              style={{ ...styles.wordCount, background: 'none', border: 'none', cursor: 'pointer' }}
              onClick={() => setText('')}
            >
              Clear
            </button>
          )}
          <button
            className="btn-primary"
            type="submit"
            disabled={loading || !text.trim() || tooShort || tooLong}
          >
            {loading ? 'Analysing…' : 'Analyse Article'}
          </button>
        </div>
      </div>
    </form>
  )
}
