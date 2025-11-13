# Mathematical Theory: Electromagnetic Induction Heating

## Table of Contents

1. [Introduction](#introduction)
2. [Electromagnetic Induction](#electromagnetic-induction)
3. [Skin Depth Effect](#skin-depth-effect)
4. [Joule Heating](#joule-heating)
5. [Fluid Dynamics](#fluid-dynamics)
6. [Thermal Efficiency](#thermal-efficiency)
7. [Numerical Implementation](#numerical-implementation)

---

## 1. Introduction

This document presents the complete mathematical derivation for electromagnetic induction heating in a cylindrical conductor with fluid flow.

### Physical System

- **Coil**: Circular AC coil with N turns, radius R_b, carrying current I(t) = I₀cos(ωt)
- **Conductor**: Cylindrical duct with inner radius a, outer radius R_out
- **Fluid**: Conductive fluid (ionized gas) with conductivity σ
- **Flow**: Axial flow with inlet velocity u_in

---

## 2. Electromagnetic Induction

### 2.1 Magnetic Field from AC Coil

For a circular coil centered at the origin, the axial magnetic field at distance z is:

```
B(z,t) = [μ₀NR_b²/(2(R_b² + z²)^(3/2))] · I(t)
```

Where:
- μ₀ = 4π×10⁻⁷ H/m (magnetic permeability of vacuum)
- N = number of turns
- R_b = coil radius [m]
- z = axial distance from coil center [m]

### 2.2 Faraday's Law of Induction

For a circular path of radius r coaxial with the coil:

```
∮ E·dl = -dΦ_B/dt
```

By symmetry, the induced electric field is tangential:

```
E_φ · 2πr = -d/dt(B · πr²)
```

Solving for E_φ:

```
E_φ(r,t) = -(r/2) · dB/dt
```

### 2.3 Time Derivative of Magnetic Field

Since I(t) = I₀cos(ωt):

```
dB/dt = [μ₀NR_b²/(2(R_b² + z²)^(3/2))] · dI/dt
      = [μ₀NR_b²/(2(R_b² + z²)^(3/2))] · (-ωI₀sin(ωt))
```

### 2.4 Induced Electric Field

Substituting into the expression for E_φ:

```
E_φ(r,z,t) = [μ₀NR_b²ωI₀/(4(R_b² + z²)^(3/2))] · r · sin(ωt)
```

**RMS Value** (for power calculations):

```
E_rms(r,z) = [μ₀NR_b²ωI₀/(4√2(R_b² + z²)^(3/2))] · r
```

**Key Insight**: The electric field grows linearly with radius r.

---

## 3. Skin Depth Effect

### 3.1 Penetration Depth

When an electromagnetic wave penetrates a conductor, its amplitude decays exponentially:

```
E(r,z_skin) = E_surface(r) · exp(-z_skin/δ)
```

Where **skin depth** δ is:

```
δ = √(2/(ωμσ))
```

Parameters:
- ω = 2πf = angular frequency [rad/s]
- μ = magnetic permeability [H/m]
- σ = electrical conductivity [S/m]

### 3.2 Physical Interpretation

- **High frequency** → small δ → heating concentrated near surface
- **High conductivity** → small δ → shallow penetration
- **Low conductivity** → large δ → volumetric heating

### 3.3 Typical Values

| Material | σ (S/m) | f (kHz) | δ (mm) |
|----------|---------|---------|--------|
| Copper | 5.8×10⁷ | 50 | 0.3 |
| Aluminum | 3.5×10⁷ | 50 | 0.4 |
| Seawater | 5 | 50 | 318 |
| Ionized H₂ | 500 | 50 | 10 |

---

## 4. Joule Heating

### 4.1 Instantaneous Power Density

The volumetric power dissipated by Joule heating is:

```
p(r,z,t) = σ · E²(r,z,t)
```

### 4.2 Time-Averaged Power Density

Using RMS values:

```
p_vol(r,z) = σ · E_rms²(r,z)
```

### 4.3 Including Skin Effect

For a conductor with finite thickness:

```
p_vol(r,z) = σ · E_rms²(r) · exp(-2z_skin/δ)
```

Where:
- z_skin = R_out - r (distance from outer wall)
- Factor of 2 appears because power ∝ E²

### 4.4 Complete Expression

Substituting E_rms:

```
p_vol(r,z) = σ · [μ₀NR_b²ωI₀/(4√2(R_b² + z²)^(3/2))]² · r² · exp(-2z_skin/δ)
```

### 4.5 Total Power per Unit Length

Integrating over the cross-section:

```
P' = ∫∫ p_vol(r,z) · r dr dφ
```

For thin skin depth (δ << a):

```
P' ≈ πaσδ · [μ₀NR_b²ωI₀/(4√2(R_b² + z²)^(3/2))]² · a²
```

Simplifying:

```
P' = [πμ₀²N²R_b⁴σδω²I₀²] / [32(R_b² + z²)³] · a³
```

---

## 5. Fluid Dynamics

### 5.1 Governing Equations (2D Axisymmetric)

In cylindrical coordinates (r, z), assuming axisymmetry (∂/∂θ = 0):

**Continuity:**
```
∂ρ/∂t + (1/r)∂(rρu_r)/∂r + ∂(ρu_z)/∂z = 0
```

**Radial Momentum:**
```
∂(ρu_r)/∂t + (1/r)∂(rρu_r²)/∂r + ∂(ρu_ru_z)/∂z = -∂p/∂r
```

**Axial Momentum:**
```
∂(ρu_z)/∂t + (1/r)∂(rρu_ru_z)/∂r + ∂(ρu_z²)/∂z = -∂p/∂z
```

**Energy:**
```
∂E_v/∂t + (1/r)∂(ru_r(E_v + p))/∂r + ∂(u_z(E_v + p))/∂z = q̇_v(r,z)
```

Where:
- ρ = density [kg/m³]
- u_r, u_z = velocity components [m/s]
- p = pressure [Pa]
- E_v = total energy density [J/m³]
- q̇_v = volumetric heat source [W/m³]

### 5.2 Equation of State (Ideal Gas)

```
p = ρRT
e = c_v T
E_v = ρ(c_v T + (u_r² + u_z²)/2)
```

Parameters:
- R = specific gas constant [J/(kg·K)]
- c_v = specific heat at constant volume [J/(kg·K)]
- γ = c_p/c_v = heat capacity ratio

### 5.3 Speed of Sound

```
a = √(γRT)
```

### 5.4 Conservative Form

Define conservative variables:
```
U = [ρ, m_r, m_z, E_v]ᵀ
```

Where:
- m_r = ρu_r (radial momentum density)
- m_z = ρu_z (axial momentum density)

The system becomes:

```
∂U/∂t + ∂F_r/∂r + ∂F_z/∂z = S
```

With flux vectors and source terms defined accordingly.

---

## 6. Thermal Efficiency

### 6.1 Definition

Thermal efficiency is the ratio of useful heating power to total input power:

```
η = P_fluid / (P_fluid + P_losses)
```

### 6.2 Components

**Useful Power (heating fluid):**
```
P_fluid = ∫∫∫ q̇_v(r,z) dV
```

**Coil Losses:**
```
P_coil = I_rms² · R_coil = (I₀²/2) · R_coil
```

**Total:**
```
η = P_fluid / (P_fluid + P_coil)
```

### 6.3 Optimization

Efficiency depends on:
- **Frequency**: Higher ω increases coupling but also skin effect
- **Conductivity**: Higher σ increases heating but decreases penetration
- **Geometry**: Optimal spacing between coil and conductor
- **Current**: Diminishing returns at very high currents

**Typical values:**
- Good design: η = 60-80%
- Excellent design: η = 80-95%
- Poor design: η < 40%

---

## 7. Numerical Implementation

### 7.1 Finite Difference Discretization

**Spatial Grid:**
```
r_i = R_in + (i - 0.5)Δr,  i = 1...N_r
z_j = (j - 0.5)Δz,         j = 1...N_z
```

**Temporal Discretization:**
CFL condition ensures stability:

```
Δt ≤ CFL · min(Δr/(|u_r| + a), Δz/(|u_z| + a))
```

Typical CFL = 0.5-0.8

### 7.2 Time Integration Scheme

**Runge-Kutta 2 (RK2):**

1. **Predictor step:**
   ```
   U* = Uⁿ + (Δt/2)·F(Uⁿ)
   ```

2. **Corrector step:**
   ```
   Uⁿ⁺¹ = Uⁿ + Δt·F(U*)
   ```

### 7.3 Boundary Conditions

**Inlet (z = 0):**
- Dirichlet: ρ = ρ_in, T = T_in, u_z = u_in
- u_r = 0

**Outlet (z = L_z):**
- Pressure: p = p_out (subsonic)
- Extrapolation: ρ, T, u from interior

**Axis (r = 0):**
- Symmetry: u_r = 0, ∂p/∂r = 0

**Wall (r = R_out):**
- Slip: u_r = 0
- Adiabatic: ∂T/∂r = 0

### 7.4 Convergence Criteria

Simulation runs until:
- t ≥ t_final, or
- Steady state: |ΔU/Δt| < ε

---

## 8. Validation Cases

### 8.1 Analytical Benchmarks

**Test 1: No flow (u = 0)**
- Should match steady-state heat equation
- Temperature profile purely diffusive

**Test 2: No heating (q̇_v = 0)**
- Should preserve inlet temperature
- Pressure drop from Euler equations

**Test 3: Mass conservation**
- Inlet mass flux = outlet mass flux
- |ṁ_in - ṁ_out|/ṁ_in < 0.1%

### 8.2 Expected Physical Behavior

✅ **Temperature increases downstream**
✅ **Peak temperature near coil location**
✅ **Exponential decay from walls (skin effect)**
✅ **Efficiency increases with conductivity**
✅ **CFL-stable for all tested cases**

---

## References

1. Jackson, J.D. (1999). *Classical Electrodynamics* (3rd ed.). Wiley.
2. Kraus, J.D. & Fleisch, D.A. (1999). *Electromagnetics* (5th ed.). McGraw-Hill.
3. Anderson, J.D. (1995). *Computational Fluid Dynamics*. McGraw-Hill.
4. Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.

---

**Document prepared for the Induction Heating Research Project**
*Last updated: 2025*
