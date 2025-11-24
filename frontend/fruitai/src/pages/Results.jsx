import React from 'react'
import Gauge from '../components/Gauge'
import DiseaseBoxesCanvas from '../components/DiseaseBoxesCanvas'

export default function Results({ result }) {
  if (!result) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">→</div>
        <p>No analysis yet</p>
        <p className="text-small">Upload an image or use your webcam to get started</p>
      </div>
    )
  }

  const { fruit, fruit_confidence, ripeness, diseases, imagePreview } = result
  const ripScore = ripeness?.ripeness_score ?? (ripeness?.fused_ripe_prob ? ripeness.fused_ripe_prob * 100 : 0)

  return (
    <div>
      <div className="card-header">
        <h2>Analysis Results</h2>
      </div>

      {/* Fruit Info */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1rem' }}>
          <div style={{
            fontSize: '2.5rem',
            opacity: 0.6
          }}>
            {fruit === 'apple' ? 'A' : fruit === 'banana' ? 'B' : fruit === 'mango' ? 'M' : '?'}
          </div>
          <div>
            <h3 style={{ marginBottom: '0.25rem' }}>{fruit ? fruit.charAt(0).toUpperCase() + fruit.slice(1) : 'Unknown'}</h3>
            <span className="badge badge-info">
              {((fruit_confidence || 0) * 100).toFixed(1)}% Confidence
            </span>
          </div>
        </div>
      </div>

      {/* Ripeness Section */}
      <div style={{ marginBottom: '2rem' }}>
        <h3 style={{ marginBottom: '1rem' }}>Ripeness Level</h3>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1.5rem' }}>
          <div style={{ width: '160px', flexShrink: 0 }}>
            <Gauge value={ripScore || 0} />
          </div>
          <div style={{ flex: 1 }}>
            <div style={{
              padding: '1rem',
              background: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border)'
            }}>
              <div style={{ marginBottom: '0.75rem' }}>
                <div className="text-small">Ripeness Score</div>
                <div style={{ fontSize: '1.75rem', fontWeight: '700', color: 'var(--primary-light)' }}>
                  {Math.round(ripScore)}%
                </div>
              </div>
              <div>
                <div className="text-small">Days to Ripen</div>
                <div style={{ fontSize: '1.25rem', fontWeight: '600' }}>
                  {ripeness?.estimated_days_left ?? '—'} days
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Disease Section */}
      <div>
        <h3 style={{ marginBottom: '1rem' }}>Disease Detection</h3>
        {diseases && diseases.length > 0 ? (
          <div>
            <DiseaseBoxesCanvas detections={diseases} imagePreview={imagePreview} />
            <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
              {diseases.map((d, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.75rem',
                    background: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: '1px solid var(--border)'
                  }}
                >
                  <div className="text-small">Detected Disease</div>
                  <div style={{ fontWeight: '600', marginBottom: '0.25rem' }}>{d.label}</div>
                  <span className="badge badge-warning">
                    {(d.confidence * 100).toFixed(1)}% Detected
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="alert alert-success">
            No diseases detected. Fruit appears healthy.
          </div>
        )}
      </div>
    </div>
  )
}