import './ResultsPanel.css'

function ResultsPanel({ results }) {
  if (!results) return null

  const lastIdx = results.times.length - 1
  const finalEfficiency = results.efficiencies[lastIdx]
  const finalMaxTemp = results.max_temperatures[lastIdx]
  const finalExitTemp = results.avg_exit_temp[lastIdx]
  const finalMaxPressure = results.max_pressures[lastIdx]

  const metrics = [
    {
      label: 'Thermal Efficiency',
      value: `${(finalEfficiency * 100).toFixed(1)}%`,
      icon: '⚡',
      color: '#10b981'
    },
    {
      label: 'Max Temperature',
      value: `${finalMaxTemp.toFixed(0)} K`,
      icon: '🌡️',
      color: '#f59e0b'
    },
    {
      label: 'Exit Temperature',
      value: `${finalExitTemp.toFixed(0)} K`,
      icon: '🔥',
      color: '#ef4444'
    },
    {
      label: 'Max Pressure',
      value: `${(finalMaxPressure / 1000).toFixed(1)} kPa`,
      icon: '💨',
      color: '#3b82f6'
    }
  ]

  return (
    <div className="results-panel">
      <h2>📊 Simulation Results</h2>
      <div className="metrics-grid">
        {metrics.map((metric, idx) => (
          <div key={idx} className="metric-card" style={{ borderColor: metric.color }}>
            <div className="metric-icon" style={{ color: metric.color }}>
              {metric.icon}
            </div>
            <div className="metric-content">
              <div className="metric-label">{metric.label}</div>
              <div className="metric-value" style={{ color: metric.color }}>
                {metric.value}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ResultsPanel
