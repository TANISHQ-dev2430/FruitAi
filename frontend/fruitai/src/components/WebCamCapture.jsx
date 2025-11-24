import React, { useRef, useState, useEffect } from 'react'
import Webcam from 'react-webcam'
import { predictYOLO } from '../services/api'

export default function WebcamCapture({ onCapture, live=false }) {
  const camRef = useRef(null)
  const overlayRef = useRef(null)
  const [loading, setLoading] = useState(false)
  const [liveActive, setLiveActive] = useState(false)
  const [detections, setDetections] = useState([])
  const [modelLoaded, setModelLoaded] = useState(true)
  const intervalRef = useRef(null)

  async function snap() {
    setLoading(true)
    try {
      const imageSrc = camRef.current.getScreenshot()
      const res = await fetch(imageSrc)
      const blob = await res.blob()
      await onCapture(new File([blob], 'webcam.jpg', { type: 'image/jpeg' }))
    } finally {
      setLoading(false)
    }
  }

  // Draw overlay boxes on canvas
  useEffect(() => {
    const canvas = overlayRef.current
    const cam = camRef.current && camRef.current.video
    if (!canvas || !cam) return
    const ctx = canvas.getContext('2d')
    // set canvas size to video size
    canvas.width = cam.videoWidth || 640
    canvas.height = cam.videoHeight || 480
    // clear
    ctx.clearRect(0,0,canvas.width,canvas.height)
    if (!detections || detections.length === 0) return
    detections.forEach(d => {
      const [x1,y1,x2,y2] = d.bbox
      const w = x2 - x1
      const h = y2 - y1
      ctx.strokeStyle = 'lime'
      ctx.lineWidth = 3
      ctx.strokeRect(x1, y1, w, h)
      ctx.fillStyle = 'rgba(0,0,0,0.6)'
      ctx.fillRect(x1, y1 - 22, ctx.measureText(d.label || '').width + 10, 20)
      ctx.fillStyle = 'white'
      ctx.font = '14px Arial'
      const labelText = `${d.label || ''} ${(d.confidence*100).toFixed(0)}%`
      ctx.fillText(labelText, x1 + 4, y1 - 6)
    })
  }, [detections])

  // Live loop: capture and send frames
  useEffect(() => {
    async function sendFrame() {
      try {
        if (!camRef.current) return
        const imageSrc = camRef.current.getScreenshot()
        if (!imageSrc) return
        const res = await fetch(imageSrc)
        const blob = await res.blob()
        const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' })
        const data = await predictYOLO(file)
        setDetections(data.detections || [])
        // reflect whether the server reports the model loaded
        if (typeof data.model_loaded !== 'undefined') setModelLoaded(Boolean(data.model_loaded))
      } catch (e) {
        console.error('Live YOLO frame failed', e)
      }
    }

    if (live && liveActive) {
      // send a frame every 600ms (~1.6 FPS). Adjust as needed.
      intervalRef.current = setInterval(sendFrame, 600)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [live, liveActive])

  return (
    <div style={{ position: 'relative' }}>
      <Webcam
        audio={false}
        ref={camRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: 'environment' }}
        style={{
          width: '100%',
          borderRadius: '12px',
          border: '2px solid var(--border)',
          marginBottom: '1.5rem',
          background: 'var(--bg-secondary)'
        }}
      />
      <canvas
        ref={overlayRef}
        style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      />

      {/* Floating detections panel (top-right) */}
      <div style={{ position: 'absolute', right: 12, top: 12, zIndex: 50 }}>
        <div style={{ minWidth: 180, background: 'rgba(0,0,0,0.6)', color: '#fff', padding: '8px 10px', borderRadius: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
            <strong style={{ fontSize: 13 }}>Live Detections</strong>
            <span style={{ fontSize: 12, opacity: 0.9 }}>{modelLoaded ? 'Model OK' : 'Model missing'}</span>
          </div>
          <div style={{ maxHeight: 160, overflowY: 'auto' }}>
            {detections && detections.length > 0 ? (
              detections.map((d, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 8, padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <div style={{ fontSize: 13 }}>{d.label}</div>
                  <div style={{ fontSize: 13, color: '#9ae6b4' }}>{(d.confidence * 100).toFixed(0)}%</div>
                </div>
              ))
            ) : (
              <div style={{ fontSize: 13, opacity: 0.85 }}>No detections</div>
            )}
          </div>
        </div>
      </div>

      {!live ? (
        <button
          onClick={snap}
          disabled={loading}
          className="btn-primary"
        >
          {loading ? (
            <>
              <span className="spinner" style={{ width: '16px', height: '16px' }} />
              Analyzing...
            </>
          ) : (
            'Capture & Analyze'
          )}
        </button>
      ) : (
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            onClick={() => setLiveActive(true)}
            disabled={liveActive}
            className="btn-primary"
          >
            Start Live
          </button>
          <button
            onClick={() => { setLiveActive(false); setDetections([]) }}
            disabled={!liveActive}
            className="btn-secondary"
          >
            Stop Live
          </button>
        </div>
      )}
    </div>
  )
}