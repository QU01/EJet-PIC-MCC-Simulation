import { useState, useEffect } from 'react'
import './SimulationControls.css'

function SimulationControls({ defaults, onRunSimulation, isRunning }) {
  const [params, setParams] = useState({
    geometry: {
      R_in: 0.02,
      R_out: 0.025,
      L_z: 0.5
    },
    coil: {
      N_coil: 100,
      I0: 500,
      frequency: 50000,
      R_coil: 0.5,
      R_b: 0.03
    },
    fluid: {
      sigma: 500,
      cp: 14300,
      cv: 10200,
      gamma: 1.41,
      R_gas: 4124
    },
    inlet: {
      pressure: 101325,
      temperature: 300,
      velocity: 50
    },
    simulation: {
      t_final: 5.0,
      CFL: 0.5,
      save_interval: 0.1,
      Nr: 30,
      Nz: 100
    }
  })

  useEffect(() => {
    if (defaults) {
      setParams(prev => ({
        ...prev,
        geometry: defaults.geometry,
        coil: defaults.coil,
        fluid: defaults.fluid,
        inlet: defaults.inlet
      }))
    }
  }, [defaults])

  const handleChange = (section, key, value) => {
    setParams(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: parseFloat(value) || 0
      }
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onRunSimulation(params)
  }

  const handleReset = () => {
    if (defaults) {
      setParams(prev => ({
        ...prev,
        geometry: defaults.geometry,
        coil: defaults.coil,
        fluid: defaults.fluid,
        inlet: defaults.inlet
      }))
    }
  }

  return (
    <div className="simulation-controls">
      <h2>⚙️ Simulation Parameters</h2>

      <form onSubmit={handleSubmit}>
        {/* Geometry Section */}
        <section className="param-section">
          <h3>📐 Geometry</h3>
          <div className="param-group">
            <label>
              Inner Radius (m)
              <input
                type="number"
                step="0.001"
                value={params.geometry.R_in}
                onChange={(e) => handleChange('geometry', 'R_in', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Outer Radius (m)
              <input
                type="number"
                step="0.001"
                value={params.geometry.R_out}
                onChange={(e) => handleChange('geometry', 'R_out', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Axial Length (m)
              <input
                type="number"
                step="0.01"
                value={params.geometry.L_z}
                onChange={(e) => handleChange('geometry', 'L_z', e.target.value)}
                disabled={isRunning}
              />
            </label>
          </div>
        </section>

        {/* Coil Section */}
        <section className="param-section">
          <h3>🔌 Coil Parameters</h3>
          <div className="param-group">
            <label>
              Number of Turns
              <input
                type="number"
                value={params.coil.N_coil}
                onChange={(e) => handleChange('coil', 'N_coil', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Peak Current (A)
              <input
                type="number"
                step="10"
                value={params.coil.I0}
                onChange={(e) => handleChange('coil', 'I0', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Frequency (Hz)
              <input
                type="number"
                step="1000"
                value={params.coil.frequency}
                onChange={(e) => handleChange('coil', 'frequency', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Coil Resistance (Ω)
              <input
                type="number"
                step="0.1"
                value={params.coil.R_coil}
                onChange={(e) => handleChange('coil', 'R_coil', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Coil Radius (m)
              <input
                type="number"
                step="0.001"
                value={params.coil.R_b}
                onChange={(e) => handleChange('coil', 'R_b', e.target.value)}
                disabled={isRunning}
              />
            </label>
          </div>
        </section>

        {/* Fluid Section */}
        <section className="param-section">
          <h3>💧 Fluid Properties</h3>
          <div className="param-group">
            <label>
              Conductivity (S/m)
              <input
                type="number"
                step="10"
                value={params.fluid.sigma}
                onChange={(e) => handleChange('fluid', 'sigma', e.target.value)}
                disabled={isRunning}
              />
            </label>
          </div>
        </section>

        {/* Inlet Section */}
        <section className="param-section">
          <h3>➡️ Inlet Conditions</h3>
          <div className="param-group">
            <label>
              Pressure (Pa)
              <input
                type="number"
                step="1000"
                value={params.inlet.pressure}
                onChange={(e) => handleChange('inlet', 'pressure', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Temperature (K)
              <input
                type="number"
                step="10"
                value={params.inlet.temperature}
                onChange={(e) => handleChange('inlet', 'temperature', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Velocity (m/s)
              <input
                type="number"
                step="5"
                value={params.inlet.velocity}
                onChange={(e) => handleChange('inlet', 'velocity', e.target.value)}
                disabled={isRunning}
              />
            </label>
          </div>
        </section>

        {/* Simulation Section */}
        <section className="param-section">
          <h3>⏱️ Simulation Settings</h3>
          <div className="param-group">
            <label>
              Final Time (s)
              <input
                type="number"
                step="0.5"
                value={params.simulation.t_final}
                onChange={(e) => handleChange('simulation', 't_final', e.target.value)}
                disabled={isRunning}
              />
            </label>
            <label>
              Grid: Radial × Axial
              <div className="grid-inputs">
                <input
                  type="number"
                  value={params.simulation.Nr}
                  onChange={(e) => handleChange('simulation', 'Nr', e.target.value)}
                  disabled={isRunning}
                  style={{ width: '48%' }}
                />
                <span>×</span>
                <input
                  type="number"
                  value={params.simulation.Nz}
                  onChange={(e) => handleChange('simulation', 'Nz', e.target.value)}
                  disabled={isRunning}
                  style={{ width: '48%' }}
                />
              </div>
            </label>
          </div>
        </section>

        <div className="button-group">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isRunning}
          >
            {isRunning ? '⏳ Running...' : '▶️ Run Simulation'}
          </button>
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleReset}
            disabled={isRunning}
          >
            🔄 Reset to Defaults
          </button>
        </div>
      </form>
    </div>
  )
}

export default SimulationControls
