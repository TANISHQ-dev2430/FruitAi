import React, { useState } from 'react'
import Home from './pages/Home'
import Results from './pages/Results'
import History from './pages/History'

export default function App() {
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [currentPage, setCurrentPage] = useState('home')

  const handleAnalysis = (analysisResult) => {
    setResult(analysisResult)
    setHistory([...history, { ...analysisResult, timestamp: new Date().toLocaleString() }])
    setCurrentPage('home')
  }

  return (
    <div style={{ minHeight: '100vh' }}>
      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="header-content">
            <div className="header-logo">F</div>
            <div>
              <h1>Fruit AI</h1>
              <p className="text-secondary">Smart ripeness & disease detection</p>
            </div>
          </div>
          <button
            onClick={() => setCurrentPage(currentPage === 'history' ? 'home' : 'history')}
            className="btn-secondary"
            style={{ padding: '0.625rem 1.25rem', fontSize: '0.875rem' }}
          >
            {currentPage === 'history' ? 'Back to Analysis' : 'History'}
          </button>
        </div>

        {/* Main Content */}
        {currentPage === 'home' ? (
          <div className="grid">
            <div>
              <div className="card">
                <Home setResult={handleAnalysis} />
              </div>
            </div>
            <div>
              <div className="card">
                <Results result={result} />
              </div>
            </div>
          </div>
        ) : (
          <div>
            <div className="card">
              <History historyItems={history} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}