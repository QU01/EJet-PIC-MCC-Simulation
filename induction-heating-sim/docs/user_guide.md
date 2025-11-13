# User Guide: Induction Heating Simulation

## Getting Started

This guide will help you set up and run your first electromagnetic induction heating simulation.

---

## Installation

### Step 1: Install Julia

Download and install Julia 1.9 or later from [julialang.org](https://julialang.org/downloads/)

Verify installation:
```bash
julia --version
```

### Step 2: Install Node.js

Download and install Node.js 18+ from [nodejs.org](https://nodejs.org/)

Verify installation:
```bash
node --version
npm --version
```

### Step 3: Setup Backend

```bash
cd induction-heating-sim/backend
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This will install all required Julia packages:
- HTTP.jl (web server)
- JSON3.jl (JSON parsing)
- LinearAlgebra, Statistics (math operations)

### Step 4: Setup Frontend

```bash
cd ../frontend
npm install
```

This installs:
- React (UI framework)
- Vite (build tool)
- Plotly.js (visualization)
- Axios (HTTP client)

---

## Running the Application

### Option 1: Using Two Terminals (Recommended)

**Terminal 1 - Backend:**
```bash
cd backend
julia --project=. src/server.jl
```

You should see:
```
======================================================================
INDUCTION HEATING SIMULATION SERVER
======================================================================
Server starting on port 8080...

Available endpoints:
  GET  /api/status   - Server status
  GET  /api/defaults - Default parameters
  POST /api/simulate - Run simulation
  GET  /api/results  - Get latest results

Press Ctrl+C to stop the server
======================================================================
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
```

**Open browser**: Navigate to `http://localhost:3000`

### Option 2: Custom Ports

If ports 3000 or 8080 are already in use:

**Backend (custom port):**
```bash
julia --project=. src/server.jl 8081
```

**Frontend (edit vite.config.js):**
```javascript
export default defineConfig({
  server: {
    port: 3001,  // Change this
    proxy: {
      '/api': {
        target: 'http://localhost:8081',  // Match backend port
        changeOrigin: true,
      }
    }
  }
})
```

---

## Using the Web Interface

### 1. Main Dashboard

When you open the application, you'll see:

```
⚡ Electromagnetic Induction Heating Simulation
2D Axisymmetric Joule Heating by Eddy Currents

[Simulation Parameters]  |  [Welcome Message]
```

### 2. Configuring Parameters

The left sidebar contains all simulation parameters:

#### 📐 Geometry
- **Inner Radius**: Fluid flow area (default: 0.02 m = 2 cm)
- **Outer Radius**: Conductor wall (default: 0.025 m = 2.5 cm)
- **Axial Length**: Total duct length (default: 0.5 m = 50 cm)

#### 🔌 Coil Parameters
- **Number of Turns**: More turns → stronger field (default: 100)
- **Peak Current**: Higher current → more heating (default: 500 A)
- **Frequency**: Affects skin depth (default: 50 kHz)
  - Low frequency (1-10 kHz): Deep penetration
  - Medium (10-100 kHz): Balanced
  - High (>100 kHz): Surface heating
- **Coil Resistance**: Affects efficiency (default: 0.5 Ω)
- **Coil Radius**: Distance from conductor (default: 0.03 m)

#### 💧 Fluid Properties
- **Conductivity**: Critical parameter (default: 500 S/m)
  - Air: ~10⁻¹⁴ S/m (not viable)
  - Ionized H₂: 100-1000 S/m (good)
  - Seawater: 5 S/m (moderate)
  - Molten metal: 10⁶-10⁷ S/m (excellent)

#### ➡️ Inlet Conditions
- **Pressure**: Starting pressure (default: 101325 Pa = 1 atm)
- **Temperature**: Inlet temperature (default: 300 K)
- **Velocity**: Flow speed (default: 50 m/s)

#### ⏱️ Simulation Settings
- **Final Time**: Physical time to simulate (default: 5 s)
- **Grid**: Resolution (default: 30×100 cells)
  - Coarse (20×50): Fast but less accurate
  - Medium (30×100): Good balance
  - Fine (50×200): Slow but detailed

### 3. Running a Simulation

1. **Adjust parameters** as desired
2. Click **▶️ Run Simulation**
3. **Wait** for completion (progress shown in console)
4. **View results** when simulation finishes

### 4. Interpreting Results

#### 📊 Metrics Panel

Four key indicators:

- **⚡ Thermal Efficiency**: Percentage of input power used for heating
  - Good: >60%
  - Excellent: >80%

- **🌡️ Max Temperature**: Highest temperature in domain
  - Should be reasonable (<2000 K for H₂)

- **🔥 Exit Temperature**: Average temperature at outlet
  - Indicates total heating effect

- **💨 Max Pressure**: Peak pressure
  - Check for compressibility effects

#### 🗺️ 2D Field Distribution

Interactive contour plot showing spatial distribution:

- **Temperature**: Hottest near coil, decays radially
- **Pressure**: Gradients drive flow
- **Axial Velocity**: Shows acceleration/deceleration
- **Density**: Decreases where temperature increases
- **Heat Source**: Shows where Joule heating occurs

**Controls:**
- Dropdown menu: Select field to visualize
- Colorbar: Shows scale and units
- Hover: See exact values at any point

#### 📈 Temporal Evolution

Three time-series plots:

1. **Thermal Efficiency**: Should stabilize quickly
2. **Temperature Evolution**:
   - Orange line: Maximum temperature (peaks near coil)
   - Red line: Exit temperature (final result)
3. **Maximum Pressure**: Should reach steady state

---

## Example Scenarios

### Scenario 1: Hydrogen Propulsion (High Altitude)

**Goal**: Heat hydrogen for atmospheric propulsion

**Parameters:**
```
Geometry: R_in=0.02m, R_out=0.025m, L_z=0.5m
Coil: N=100, I0=500A, f=50kHz, R_coil=0.5Ω
Fluid: σ=500 S/m (preionized H₂)
Inlet: p=10kPa (high altitude), T=200K, v=100m/s
```

**Expected Results:**
- Efficiency: 65-75%
- Exit temp: 350-400 K
- ΔT: ~150 K

### Scenario 2: Industrial Heating (Molten Salt)

**Goal**: Heat conductive molten salt

**Parameters:**
```
Geometry: R_in=0.05m, R_out=0.055m, L_z=1.0m
Coil: N=200, I0=1000A, f=10kHz, R_coil=1.0Ω
Fluid: σ=5000 S/m (molten salt)
Inlet: p=101kPa, T=600K, v=10m/s
```

**Expected Results:**
- Efficiency: 80-90%
- Exit temp: 800-900 K
- ΔT: 200-300 K

### Scenario 3: Optimization Study

**Goal**: Find optimal frequency

**Method:**
1. Keep all parameters fixed except frequency
2. Run simulations for f = [1, 5, 10, 50, 100, 200] kHz
3. Plot efficiency vs. frequency
4. Identify optimal frequency

---

## Troubleshooting

### Problem: Simulation crashes or gives NaN

**Causes:**
- CFL condition violated (timestep too large)
- Unphysical parameters (negative values)
- Extreme temperature gradients

**Solutions:**
- Reduce CFL number (try 0.3 instead of 0.5)
- Increase grid resolution
- Check parameter values

### Problem: Efficiency is very low (<20%)

**Causes:**
- Conductivity too low
- Coil resistance too high
- Poor geometric coupling

**Solutions:**
- Increase fluid conductivity
- Reduce coil resistance
- Move coil closer to conductor

### Problem: Temperature doesn't increase

**Causes:**
- Zero or very low conductivity
- Coil current too low
- Flow velocity too high (convection dominates)

**Solutions:**
- Check σ > 100 S/m
- Increase I₀
- Reduce inlet velocity

### Problem: Backend won't start

**Error:** Package not found

**Solution:**
```bash
cd backend
rm -rf Manifest.toml
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Problem: Frontend won't connect to backend

**Error:** Network error / Failed to fetch

**Solutions:**
1. Verify backend is running (`http://localhost:8080/api/status`)
2. Check browser console for CORS errors
3. Ensure proxy is configured in `vite.config.js`

---

## Advanced Usage

### Saving Results

Results are automatically stored in memory. To save to disk:

**Option 1: Use API**
```bash
curl http://localhost:8080/api/results > results.json
```

**Option 2: Modify Julia code**

Edit `simulation.jl`:
```julia
# After run_simulation
save_results(results, "my_simulation.json")
```

### Batch Simulations

Create a script to run multiple cases:

```julia
# batch_run.jl
using JSON3

params_list = [
    Dict("freq" => 10e3, "I0" => 300),
    Dict("freq" => 50e3, "I0" => 500),
    Dict("freq" => 100e3, "I0" => 700),
]

for (i, params) in enumerate(params_list)
    println("Running case $i...")
    # Modify and run simulation
    # Save results
end
```

### Custom Physics

To modify the physics model:

1. Edit `backend/src/physics.jl` (equations)
2. Edit `backend/src/solver.jl` (numerical methods)
3. Restart backend server

---

## Performance Tips

### For Faster Simulations

1. **Reduce grid resolution**: 20×50 instead of 30×100
2. **Shorter simulation time**: 2-3 seconds often sufficient
3. **Use Julia 1.10+**: Latest version has performance improvements
4. **Compile backend**: Precompile packages on first run

### For More Accurate Results

1. **Increase grid resolution**: 50×200 or higher
2. **Reduce CFL**: 0.3 for better stability
3. **Longer simulation**: Run until steady state (check convergence)
4. **Second-order schemes**: Modify solver for MUSCL or WENO

---

## Best Practices

✅ **Always check mass conservation**: Inlet mass flux ≈ outlet mass flux
✅ **Monitor CFL condition**: Should stay below 1.0
✅ **Verify physical limits**: Temperatures, pressures should be reasonable
✅ **Start with defaults**: Modify one parameter at a time
✅ **Compare with theory**: Use analytical estimates for validation

---

## Getting Help

- **Documentation**: See `docs/theory.md` for mathematical details
- **Issues**: Open a GitHub issue for bugs or questions
- **Examples**: Check `examples/` directory (if available)
- **Community**: Join discussions on project forum

---

## Next Steps

After mastering the basics:

1. **Explore parameter space**: Systematic sensitivity studies
2. **Validate results**: Compare with experimental data or literature
3. **Extend the model**: Add viscosity, radiation, chemical reactions
4. **Publish findings**: Use results for research papers

---

**Happy simulating! ⚡**
