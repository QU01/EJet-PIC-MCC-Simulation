import { useState } from 'react'
import Plot from 'react-plotly.js'
import './Visualizations.css'

function Visualizations({ results }) {
  const [selectedField, setSelectedField] = useState('temperature')

  if (!results) return null

  const { grid, final_state } = results

  // Prepare data for 2D contour plot
  const fieldData = {
    temperature: {
      z: final_state.temperature,
      label: 'Temperature (K)',
      colorscale: 'Hot'
    },
    pressure: {
      z: final_state.pressure.map(row => row.map(val => val / 1000)),
      label: 'Pressure (kPa)',
      colorscale: 'Viridis'
    },
    velocity_z: {
      z: final_state.velocity_z,
      label: 'Axial Velocity (m/s)',
      colorscale: 'Blues'
    },
    density: {
      z: final_state.density,
      label: 'Density (kg/m³)',
      colorscale: 'YlOrRd'
    },
    heat_source: {
      z: final_state.heat_source.map(row => row.map(val => val / 1e6)),
      label: 'Heat Source (MW/m³)',
      colorscale: 'Reds'
    }
  }

  const currentField = fieldData[selectedField]

  // Convert to meters for display
  const r_mm = grid.r.map(val => val * 1000)
  const z_mm = grid.z.map(val => val * 1000)

  // Temporal evolution data
  const timeData = {
    x: results.times,
    efficiency: results.efficiencies.map(val => val * 100),
    maxTemp: results.max_temperatures,
    exitTemp: results.avg_exit_temp,
    maxPressure: results.max_pressures.map(val => val / 1000)
  }

  return (
    <div className="visualizations">
      <div className="viz-section">
        <div className="section-header">
          <h2>🗺️ 2D Field Distribution</h2>
          <div className="field-selector">
            <label>Field: </label>
            <select
              value={selectedField}
              onChange={(e) => setSelectedField(e.target.value)}
            >
              <option value="temperature">Temperature</option>
              <option value="pressure">Pressure</option>
              <option value="velocity_z">Axial Velocity</option>
              <option value="density">Density</option>
              <option value="heat_source">Heat Source</option>
            </select>
          </div>
        </div>

        <div className="plot-container">
          <Plot
            data={[
              {
                x: z_mm,
                y: r_mm,
                z: currentField.z,
                type: 'contour',
                colorscale: currentField.colorscale,
                contours: {
                  coloring: 'heatmap'
                },
                colorbar: {
                  title: currentField.label,
                  thickness: 20,
                  len: 0.7
                }
              }
            ]}
            layout={{
              title: `${currentField.label} Distribution`,
              xaxis: {
                title: 'Axial Position (mm)',
                gridcolor: '#e5e5e5'
              },
              yaxis: {
                title: 'Radial Position (mm)',
                gridcolor: '#e5e5e5'
              },
              autosize: true,
              paper_bgcolor: 'white',
              plot_bgcolor: 'white',
              font: { family: 'inherit' }
            }}
            config={{ responsive: true }}
            style={{ width: '100%', height: '500px' }}
          />
        </div>
      </div>

      <div className="viz-section">
        <h2>📈 Temporal Evolution</h2>
        <div className="plots-grid">
          <div className="plot-container">
            <Plot
              data={[
                {
                  x: timeData.x,
                  y: timeData.efficiency,
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#10b981', width: 3 },
                  marker: { size: 6 },
                  name: 'Efficiency'
                }
              ]}
              layout={{
                title: 'Thermal Efficiency',
                xaxis: { title: 'Time (s)' },
                yaxis: { title: 'Efficiency (%)' },
                autosize: true,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                font: { family: 'inherit' }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '300px' }}
            />
          </div>

          <div className="plot-container">
            <Plot
              data={[
                {
                  x: timeData.x,
                  y: timeData.maxTemp,
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#f59e0b', width: 3 },
                  marker: { size: 6 },
                  name: 'Max Temp'
                },
                {
                  x: timeData.x,
                  y: timeData.exitTemp,
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#ef4444', width: 3 },
                  marker: { size: 6 },
                  name: 'Exit Temp'
                }
              ]}
              layout={{
                title: 'Temperature Evolution',
                xaxis: { title: 'Time (s)' },
                yaxis: { title: 'Temperature (K)' },
                autosize: true,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                font: { family: 'inherit' },
                showlegend: true,
                legend: { x: 0.7, y: 1 }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '300px' }}
            />
          </div>

          <div className="plot-container">
            <Plot
              data={[
                {
                  x: timeData.x,
                  y: timeData.maxPressure,
                  type: 'scatter',
                  mode: 'lines+markers',
                  line: { color: '#3b82f6', width: 3 },
                  marker: { size: 6 },
                  name: 'Max Pressure'
                }
              ]}
              layout={{
                title: 'Maximum Pressure',
                xaxis: { title: 'Time (s)' },
                yaxis: { title: 'Pressure (kPa)' },
                autosize: true,
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                font: { family: 'inherit' }
              }}
              config={{ responsive: true }}
              style={{ width: '100%', height: '300px' }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Visualizations
