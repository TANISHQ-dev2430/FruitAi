import React, { useRef, useState } from 'react'

export default function FileUpload({ onFile, onAnalyze, loading }) {
  const ref = useRef()
  const [fileName, setFileName] = useState(null)
  const [isDragging, setIsDragging] = useState(false)

  function handleChange(e) {
    const f = e.target.files[0]
    if (!f) return
    setFileName(f.name)
    onFile(f)
  }

  function handleDragOver(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  function handleDragLeave(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  function handleDrop(e) {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/')) {
      setFileName(f.name)
      onFile(f)
    }
  }

  function handleLabelClick() {
    ref.current?.click()
  }

  return (
    <div className="file-upload">
      {/* File Input Area */}
      <div
        onClick={handleLabelClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '3rem 2rem',
          border: isDragging ? '2px solid var(--primary)' : '2px dashed var(--border)',
          borderRadius: '12px',
          cursor: 'pointer',
          transition: 'all 0.3s ease',
          backgroundColor: isDragging ? 'rgba(79, 70, 229, 0.1)' : 'var(--bg-secondary)',
          marginBottom: '1.5rem'
        }}
      >
        <input
          ref={ref}
          id="file-input"
          type="file"
          accept="image/*"
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem', fontWeight: '700', color: 'var(--primary-light)' }}>↑</div>
        <div style={{ fontWeight: '700', marginBottom: '0.5rem', color: 'var(--text-primary)', fontSize: '1.1rem' }}>Choose an image or drag & drop</div>
        <div className="text-small">PNG, JPG or GIF (max. 10MB)</div>
      </div>

      {/* Selected File */}
      {fileName && (
        <div style={{
          padding: '0.75rem 1rem',
          background: 'rgba(16, 185, 129, 0.15)',
          border: '1px solid rgba(16, 185, 129, 0.4)',
          borderRadius: '8px',
          marginBottom: '1rem',
          color: 'var(--success)',
          fontSize: '0.875rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          ✓ {fileName}
        </div>
      )}

      {/* Analyze Button (centered) */}
      <div className="analyze-row" style={{ display: 'flex', justifyContent: 'center' }}>
        <button
          onClick={() => {
            onAnalyze()
            setFileName(null)
          }}
          disabled={!fileName || loading}
          className="btn-primary"
          style={{ minWidth: 300 }}
        >
          {loading ? (
            <>
              <span className="spinner" style={{ width: '18px', height: '18px' }} />
              Analyzing...
            </>
          ) : (
            'Analyze Image'
          )}
        </button>
      </div>
    </div>
  )
}