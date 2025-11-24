import React, { useState } from 'react'
import FileUpload from '../components/FileUpload'
import WebcamCapture from '../components/WebcamCapture'
import { predictFruit, predictRipeness, predictDisease } from '../services/api'

export default function Home({ setResult }) {
  const [file, setFile] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [mode, setMode] = useState('upload')
  const [loading, setLoading] = useState(false)

  async function handleAnalyze(uploadedFile) {
    setLoading(true)
    try {
      // Create image preview for display - wait for it to load before proceeding
      const imageData = await new Promise((resolve) => {
        const reader = new FileReader()
        reader.onload = (e) => {
          resolve(e.target.result)
        }
        reader.readAsDataURL(uploadedFile)
      })
      setImagePreview(imageData)

      const fruitRes = await predictFruit(uploadedFile)
      const fruit = fruitRes.fruit || fruitRes.label || 'unknown'
      const rip = await predictRipeness(uploadedFile, fruit)
      const disease = await predictDisease(uploadedFile, fruit)
      const out = {
        fruit,
        fruit_confidence: fruitRes.fruit_confidence || fruitRes.confidence || 1.0,
        ripeness: rip,
        diseases: disease.detections || [],
        imagePreview: imageData
      }
      setResult(out)
      // Reset file for next upload
      setFile(null)
    } catch (e) {
      console.error(e)
      alert('Analysis failed. Make sure the backend is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <div className="card-header">
        <h2>Upload & Analyze</h2>
        {/* Icon Toggle Buttons - moved into card-header (top-right) */}
        <div className="icon-toggle" style={{ marginLeft: 'auto' }}>
          <button
            className={`icon-btn ${mode === 'webcam' ? 'active' : ''}`}
            onClick={() => setMode('webcam')}
            aria-label="Open webcam"
            title="Open webcam"
          >
            {/* Camera SVG */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M4 7H6L7 5H17L18 7H20C21.1046 7 22 7.89543 22 9V19C22 20.1046 21.1046 21 20 21H4C2.89543 21 2 20.1046 2 19V9C2 7.89543 2.89543 7 4 7Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="14" r="3" stroke="currentColor" strokeWidth="1.5"/>
            </svg>
          </button>

          <button
            className={`icon-btn ${mode === 'upload' ? 'active' : ''}`}
            onClick={() => setMode('upload')}
            aria-label="Open file upload"
            title="Open upload"
          >
            {/* Folder SVG */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M3 7C3 5.89543 3.89543 5 5 5H9L11 7H19C20.1046 7 21 7.89543 21 9V19C21 20.1046 20.1046 21 19 21H5C3.89543 21 3 20.1046 3 19V7Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button
            className={`icon-btn ${mode === 'live' ? 'active' : ''}`}
            onClick={() => setMode('live')}
            aria-label="Live detection"
            title="Live detection"
            style={{ marginLeft: '0.5rem' }}
          >
            {/* Play / Live SVG */}
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden>
              <path d="M5 3v18l15-9L5 3z" fill="currentColor"/>
            </svg>
          </button>
        </div>

      </div>

      {/* Upload Box container (visual wrapper) */}
      <div style={{ 
        background: 'transparent',
        border: 'none',
        borderRadius: '12px',
        padding: '0',
        position: 'relative'
      }}>

        {/* Content */}
        <div style={{ marginTop: '0rem' }}>
          {mode === 'upload' && (
            <FileUpload
              onFile={(f) => setFile(f)}
              onAnalyze={() => file && handleAnalyze(file)}
              loading={loading}
            />
          )}

          {mode === 'webcam' && (
            <WebcamCapture
              onCapture={async (blob) => {
                await handleAnalyze(blob)
              }}
            />
          )}

          {mode === 'live' && (
            <WebcamCapture
              live={true}
              onCapture={async (blob) => {
                // keep existing upload analyze behavior when user manually captures
                await handleAnalyze(blob)
              }}
            />
          )}
        </div>
      </div>
    </div>
  )
}