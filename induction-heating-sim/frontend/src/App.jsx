import { useState, useEffect } from 'react'
import SimulationControls from './components/SimulationControls'
import Visualizations from './components/Visualizations'
import ResultsPanel from './components/ResultsPanel'
import './App.css'

const API_BASE = 'http://localhost:8080'

function App() {
  const [simulationState, setSimulationState] = useState({
    isRunning: false,
    hasResults: false,
    error: null
  })

  const [results, setResults] = useState(null)
  const [defaults, setDefaults] = useState(null)

  // Load default parameters on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/defaults`)
      .then(res => res.json())
      .then(data => setDefaults(data))
      .catch(err => console.error('Error loading defaults:', err))
  }, [])

  const runSimulation = async (params) => {
    setSimulationState({ isRunning: true, hasResults: false, error: null })
    setResults(null)

    try {
      const response = await fetch(`${API_BASE}/api/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.message || 'Simulation failed')
      }

      const data = await response.json()
      setResults(data)
      setSimulationState({ isRunning: false, hasResults: true, error: null })
    } catch (error) {
      setSimulationState({
        isRunning: false,
        hasResults: false,
        error: error.message
      })
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>⚡ Electromagnetic Induction Heating Simulation</h1>
          <p className="subtitle">
            2D Axisymmetric Joule Heating by Eddy Currents
          </p>
        </div>
      </header>

      <main className="app-main">
        <div className="layout">
          <aside className="sidebar">
            <SimulationControls
              defaults={defaults}
              onRunSimulation={runSimulation}
              isRunning={simulationState.isRunning}
            />
          </aside>

          <section className="content">
            {simulationState.error && (
              <div className="error-banner">
                <strong>Error:</strong> {simulationState.error}
              </div>
            )}

            {simulationState.isRunning && (
              <div className="loading-banner">
                <div className="spinner"></div>
                <p>Running simulation... This may take a few moments.</p>
              </div>
            )}

            {simulationState.hasResults && results && (
              <>
                <ResultsPanel results={results} />
                <Visualizations results={results} />
              </>
            )}

            {!simulationState.hasResults && !simulationState.isRunning && (
              <div className="welcome-message">
                <h2>Welcome to the Induction Heating Simulator</h2>
                <p>
                  Configure your simulation parameters on the left and click
                  <strong> Run Simulation</strong> to begin.
                </p>
                <div className="features">
                  <div className="feature">
                    <span className="icon">🔬</span>
                    <h3>Physics-Based</h3>
                    <p>Accurate electromagnetic induction modeling</p>
                  </div>
                  <div className="feature">
                    <span className="icon">📊</span>
                    <h3>Interactive Visualization</h3>
                    <p>Real-time 2D contour plots and graphs</p>
                  </div>
                  <div className="feature">
                    <span className="icon">⚙️</span>
                    <h3>Customizable</h3>
                    <p>Adjust coil, fluid, and geometry parameters</p>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      <footer className="app-footer">
        <p>© 2025 Induction Heating Research Project | Powered by Julia & React</p>
      </footer>
    </div>
  )
}

export default App
