import React, { useState } from 'react'

export default function History({ historyItems }) {
  const [selectedItem, setSelectedItem] = useState(null)

  if (!historyItems || historyItems.length === 0) {
    return (
      <div>
        <div className="card-header">
          <h2>Analysis History</h2>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <p>No analysis history yet</p>
          <p className="text-small">Your analysis results will appear here</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div className="card-header">
        <h2>Analysis History ({historyItems.length})</h2>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
        {/* History List */}
        <div>
          <h3 style={{ marginBottom: '1rem' }}>Recent Analyses</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {[...historyItems].reverse().map((item, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedItem(item)}
                style={{
                  padding: '1rem',
                  background: selectedItem === item ? 'rgba(79, 70, 229, 0.15)' : 'var(--bg-secondary)',
                  border: selectedItem === item ? '1px solid var(--primary)' : '1px solid var(--border)',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
                      {item.fruit ? item.fruit.charAt(0).toUpperCase() + item.fruit.slice(1) : 'Unknown'}
                    </div>
                    <div className="text-small">{item.timestamp}</div>
                  </div>
                  <span className="badge badge-info">
                    {((item.fruit_confidence || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Details Panel */}
        <div>
          {selectedItem ? (
            <div>
              <h3 style={{ marginBottom: '1rem' }}>Details</h3>
              
              {/* Fruit Info */}
              <div style={{
                padding: '1rem',
                background: 'var(--bg-secondary)',
                borderRadius: '8px',
                border: '1px solid var(--border)',
                marginBottom: '1rem'
              }}>
                <div className="text-small">Fruit</div>
                <h4 style={{ marginBottom: '0.5rem', fontSize: '1.5rem', fontWeight: '700' }}>
                  {selectedItem.fruit ? selectedItem.fruit.charAt(0).toUpperCase() + selectedItem.fruit.slice(1) : 'Unknown'}
                </h4>
                <span className="badge badge-info">
                  {((selectedItem.fruit_confidence || 0) * 100).toFixed(1)}% Confidence
                </span>
              </div>

              {/* Ripeness */}
              {selectedItem.ripeness && (
                <div style={{
                  padding: '1rem',
                  background: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  marginBottom: '1rem'
                }}>
                  <div className="text-small">Ripeness Score</div>
                  <div style={{ fontSize: '2rem', fontWeight: '700', color: 'var(--primary-light)', marginBottom: '0.5rem' }}>
                    {selectedItem.ripeness.ripeness_score ? Math.round(selectedItem.ripeness.ripeness_score) : 'N/A'}%
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div>
                      <div className="text-small">Status</div>
                      <div style={{ fontWeight: '600' }}>
                        {selectedItem.ripeness.label ? selectedItem.ripeness.label.charAt(0).toUpperCase() + selectedItem.ripeness.label.slice(1) : '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-small">Days Left</div>
                      <div style={{ fontWeight: '600' }}>
                        {selectedItem.ripeness.estimated_days_left ?? '—'}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Diseases */}
              {selectedItem.diseases && selectedItem.diseases.length > 0 && (
                <div style={{
                  padding: '1rem',
                  background: 'var(--bg-secondary)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)',
                  marginBottom: '1rem'
                }}>
                  <div className="text-small">Diseases Detected</div>
                  <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {selectedItem.diseases.map((d, i) => (
                      <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: '500' }}>{d.label}</span>
                        <span className="badge badge-warning">
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(!selectedItem.diseases || selectedItem.diseases.length === 0) && (
                <div className="alert alert-success">
                  No diseases detected
                </div>
              )}

              {/* Timestamp */}
              <div style={{
                padding: '0.75rem',
                color: 'var(--text-secondary)',
                fontSize: '0.875rem',
                textAlign: 'center',
                borderTop: '1px solid var(--border)',
                marginTop: '1rem',
                paddingTop: '1rem'
              }}>
                Analyzed on {selectedItem.timestamp}
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">→</div>
              <p>Select an analysis to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
