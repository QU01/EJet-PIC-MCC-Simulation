# ⚡ Electromagnetic Induction Heating Simulation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Julia](https://img.shields.io/badge/Julia-1.9+-9558B2.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)

A comprehensive research project for simulating Joule heating by electromagnetic induction in cylindrical conductors. Features a high-performance Julia backend and an interactive web-based visualization interface.

## 🎯 Overview

This project implements a 2D axisymmetric finite difference simulation of electromagnetic induction heating in a cylindrical fluid conductor. The system models:

- **Electromagnetic field induction** from an AC coil
- **Eddy current generation** in conductive fluid
- **Joule heating** with skin depth effects
- **Compressible inviscid fluid dynamics**
- **Real-time thermal efficiency calculation**

### Key Features

- 🔬 **Physics-accurate**: Full derivation from Maxwell's equations
- 🚀 **High-performance**: Julia backend for fast computation
- 📊 **Interactive visualization**: Real-time 2D contour plots and temporal graphs
- ⚙️ **Configurable**: Adjust coil, fluid, geometry, and simulation parameters
- 🌐 **Web-based**: Modern React interface with responsive design

## 📐 Mathematical Model

The simulation solves the coupled system of equations:

### 1. Electromagnetic Induction

Electric field induced by AC coil:
```
E_rms(r,z) = [μ₀NR_b²ωI₀/(4√2(R_b² + z²)^(3/2))] · r
```

Skin depth:
```
δ = √(2/(ωμσ))
```

### 2. Volumetric Heating

Joule power density with skin effect:
```
q̇_v(r,z) = σE_rms² · exp(-2z_skin/δ)
```

### 3. Fluid Dynamics (2D Axisymmetric)

Conservation equations in cylindrical coordinates:
```
∂ρ/∂t + (1/r)∂(rm_r)/∂r + ∂m_z/∂z = 0

∂m_r/∂t + ... = -∂p/∂r

∂m_z/∂t + ... = -∂p/∂z

∂E_v/∂t + ... = q̇_v(r,z)
```

### 4. Thermal Efficiency

```
η = P_fluid / (P_fluid + P_coil)
```

See [docs/theory.md](docs/theory.md) for complete mathematical derivation.

## 🏗️ Project Structure

```
induction-heating-sim/
├── backend/                    # Julia simulation engine
│   ├── src/
│   │   ├── physics.jl         # Physical equations
│   │   ├── solver.jl          # Finite difference solver
│   │   ├── simulation.jl      # Main simulation orchestrator
│   │   └── server.jl          # HTTP API server
│   └── Project.toml           # Julia dependencies
│
├── frontend/                   # React web interface
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── SimulationControls.jsx
│   │   │   ├── Visualizations.jsx
│   │   │   └── ResultsPanel.jsx
│   │   ├── App.jsx            # Main app component
│   │   └── main.jsx           # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── docs/                       # Documentation
│   ├── theory.md              # Mathematical theory
│   └── user_guide.md          # Usage instructions
│
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Julia 1.9+** ([Download](https://julialang.org/downloads/))
- **Node.js 18+** ([Download](https://nodejs.org/))
- **npm** (comes with Node.js)

### Installation

1. **Clone the repository**
   ```bash
   cd induction-heating-sim
   ```

2. **Setup Julia backend**
   ```bash
   cd backend
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

3. **Setup React frontend**
   ```bash
   cd ../frontend
   npm install
   ```

### Running the Application

1. **Start the Julia backend server** (Terminal 1)
   ```bash
   cd backend
   julia --project=. src/server.jl
   ```
   Server will start on `http://localhost:8080`

2. **Start the React frontend** (Terminal 2)
   ```bash
   cd frontend
   npm run dev
   ```
   Frontend will start on `http://localhost:3000`

3. **Open your browser**
   Navigate to `http://localhost:3000`

## 📊 Usage

### Web Interface

1. **Configure Parameters**
   - Adjust geometry (radius, length)
   - Set coil parameters (turns, current, frequency)
   - Define fluid properties (conductivity)
   - Set inlet conditions (pressure, temperature, velocity)
   - Choose simulation settings (time, grid resolution)

2. **Run Simulation**
   - Click "Run Simulation"
   - Monitor progress in console
   - View results when complete

3. **Analyze Results**
   - **Metrics Panel**: View key performance indicators
   - **2D Contours**: Visualize temperature, pressure, velocity fields
   - **Temporal Plots**: Track efficiency and temperatures over time

### API Endpoints

The Julia backend exposes a REST API:

- `GET /api/status` - Server status
- `GET /api/defaults` - Default parameter values
- `POST /api/simulate` - Run simulation with custom parameters
- `GET /api/results` - Retrieve latest simulation results

### Example API Request

```bash
curl -X POST http://localhost:8080/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "geometry": {"R_in": 0.02, "R_out": 0.025, "L_z": 0.5},
    "coil": {"N_coil": 100, "I0": 500, "frequency": 50000},
    "fluid": {"sigma": 500},
    "inlet": {"pressure": 101325, "temperature": 300, "velocity": 50},
    "simulation": {"t_final": 5.0, "Nr": 30, "Nz": 100}
  }'
```

## 🔬 Research Applications

This simulation framework is suitable for studying:

1. **Aerospace propulsion**
   - Electromagnetic thermal propulsion systems
   - High-altitude plasma heating
   - Atmospheric flight on other planets (e.g., Saturn)

2. **Industrial heating**
   - Induction furnaces
   - Plasma torches
   - RF heating systems

3. **Energy conversion**
   - MHD generators
   - Plasma-assisted combustion
   - Thermal efficiency optimization

## 📈 Performance

Typical simulation performance (30×100 grid, 5 seconds physical time):

- **Computation time**: 2-5 minutes (depending on CPU)
- **Memory usage**: ~500 MB
- **Convergence**: CFL-limited timestep ensures stability

## 🧪 Example Results

### Hydrogen Flow with Preionization

**Parameters:**
- Inner radius: 2 cm
- Coil: 100 turns, 500 A peak, 50 kHz
- Fluid: σ = 500 S/m (preionized H₂)
- Inlet: 300 K, 1 atm, 50 m/s

**Expected Results:**
- Thermal efficiency: 60-70%
- Exit temperature rise: 150-200 K
- Maximum temperature: 450-500 K

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 References

1. **Maxwell's Equations and Electromagnetic Induction**
   - Jackson, J.D. "Classical Electrodynamics" (3rd ed.)

2. **Skin Effect and Eddy Currents**
   - Kraus, J.D. "Electromagnetics" (4th ed.)

3. **Computational Fluid Dynamics**
   - Anderson, J.D. "Computational Fluid Dynamics"

4. **Finite Difference Methods**
   - LeVeque, R.J. "Finite Difference Methods for ODEs and PDEs"

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Research Team - *Initial work*

## 🙏 Acknowledgments

- Julia community for excellent scientific computing tools
- React and Plotly.js for powerful visualization capabilities
- All contributors to the open-source scientific software ecosystem

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Made with ⚡ and ❤️ for computational physics research**
