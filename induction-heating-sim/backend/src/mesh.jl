"""
Mesh Module: Non-uniform grid generation with wall refinement
Implements various stretching functions for clustering near boundaries
"""
module Mesh

export Grid, create_stretched_grid, clustering_function

"""
Stretching functions for non-uniform mesh generation
"""

"""
    tanh_stretching(η, β)

Hyperbolic tangent stretching function
- η: uniform coordinate [0, 1]
- β: stretching parameter (β > 0, larger = more clustering)
Returns stretched coordinate ξ ∈ [0, 1]
"""
function tanh_stretching(η::Float64, β::Float64)::Float64
    if β < 1e-6
        return η  # No stretching
    end

    numerator = tanh(β * (η - 0.5))
    denominator = tanh(0.5 * β)

    return 0.5 * (1.0 + numerator / denominator)
end

"""
    exponential_stretching(η, r)

Exponential stretching function
- η: uniform coordinate [0, 1]
- r: stretching ratio (r > 1, larger = more clustering at η=0)
Returns stretched coordinate
"""
function exponential_stretching(η::Float64, r::Float64)::Float64
    if abs(r - 1.0) < 1e-6
        return η
    end

    return (r^η - 1.0) / (r - 1.0)
end

"""
    double_sided_stretching(η, β_start, β_end)

Two-sided stretching for clustering at both ends
- η: uniform coordinate [0, 1]
- β_start: clustering at η=0
- β_end: clustering at η=1
"""
function double_sided_stretching(η::Float64, β_start::Float64, β_end::Float64)::Float64
    # Combine two exponential stretchings
    if η < 0.5
        η_local = 2.0 * η
        ξ_local = exponential_stretching(η_local, β_start)
        return 0.5 * ξ_local
    else
        η_local = 2.0 * (1.0 - η)
        ξ_local = exponential_stretching(η_local, β_end)
        return 1.0 - 0.5 * ξ_local
    end
end

"""
Grid structure with support for non-uniform spacing
"""
struct Grid
    Nr::Int                    # Number of radial cells
    Nz::Int                    # Number of axial cells
    r::Vector{Float64}         # Radial cell centers [m]
    z::Vector{Float64}         # Axial cell centers [m]
    r_faces::Vector{Float64}   # Radial cell faces [m]
    z_faces::Vector{Float64}   # Axial cell faces [m]
    dr::Vector{Float64}        # Radial cell sizes [m]
    dz::Vector{Float64}        # Axial cell sizes [m]
    uniform::Bool              # Is grid uniform?

    # Metrics for cylindrical coordinates
    dr_min::Float64
    dr_max::Float64
    dz_min::Float64
    dz_max::Float64
end

"""
    create_uniform_grid(R_in, R_out, L_z, Nr, Nz)

Create uniform grid (original implementation)
"""
function create_uniform_grid(R_in::Float64, R_out::Float64, L_z::Float64,
                            Nr::Int, Nz::Int)
    dr_uniform = (R_out - R_in) / Nr
    dz_uniform = L_z / Nz

    # Cell faces
    r_faces = [R_in + i * dr_uniform for i in 0:Nr]
    z_faces = [j * dz_uniform for j in 0:Nz]

    # Cell centers
    r = [0.5 * (r_faces[i] + r_faces[i+1]) for i in 1:Nr]
    z = [0.5 * (z_faces[j] + z_faces[j+1]) for j in 1:Nz]

    # Cell sizes
    dr = fill(dr_uniform, Nr)
    dz = fill(dz_uniform, Nz)

    Grid(Nr, Nz, r, z, r_faces, z_faces, dr, dz, true,
         dr_uniform, dr_uniform, dz_uniform, dz_uniform)
end

"""
    create_stretched_grid(R_in, R_out, L_z, Nr, Nz;
                         radial_stretching=2.0, axial_stretching=1.0)

Create non-uniform grid with clustering near walls

Parameters:
- R_in, R_out: Inner and outer radii [m]
- L_z: Axial length [m]
- Nr, Nz: Number of cells
- radial_stretching: Clustering ratio near outer wall (r=R_out)
  - 1.0 = uniform
  - 1.5-2.5 = moderate clustering
  - 3.0+ = strong clustering
- axial_stretching: Clustering ratio near outlet (z=L_z)
"""
function create_stretched_grid(R_in::Float64, R_out::Float64, L_z::Float64,
                              Nr::Int, Nz::Int;
                              radial_stretching::Float64=2.0,
                              axial_stretching::Float64=1.0)

    # Radial direction (cluster near outer wall for boundary layer)
    r_faces = zeros(Nr + 1)
    r_faces[1] = R_in
    r_faces[end] = R_out

    for i in 2:Nr
        η = (i - 1) / Nr  # Uniform coordinate [0, 1]
        ξ = exponential_stretching(η, radial_stretching)
        r_faces[i] = R_in + ξ * (R_out - R_in)
    end

    # Axial direction (optional clustering)
    z_faces = zeros(Nz + 1)
    z_faces[1] = 0.0
    z_faces[end] = L_z

    if axial_stretching > 1.01
        for j in 2:Nz
            η = (j - 1) / Nz
            ξ = exponential_stretching(η, axial_stretching)
            z_faces[j] = ξ * L_z
        end
    else
        # Uniform in axial direction
        for j in 2:Nz
            z_faces[j] = (j - 1) * L_z / Nz
        end
    end

    # Cell centers
    r = [0.5 * (r_faces[i] + r_faces[i+1]) for i in 1:Nr]
    z = [0.5 * (z_faces[j] + z_faces[j+1]) for j in 1:Nz]

    # Cell sizes
    dr = [r_faces[i+1] - r_faces[i] for i in 1:Nr]
    dz = [z_faces[j+1] - z_faces[j] for j in 1:Nz]

    # Metrics
    dr_min, dr_max = extrema(dr)
    dz_min, dz_max = extrema(dz)

    Grid(Nr, Nz, r, z, r_faces, z_faces, dr, dz, false,
         dr_min, dr_max, dz_min, dz_max)
end

"""
    print_grid_info(grid)

Print grid statistics
"""
function print_grid_info(grid::Grid)
    println("Grid Information:")
    println("  Type: ", grid.uniform ? "Uniform" : "Non-uniform (stretched)")
    println("  Size: $(grid.Nr) × $(grid.Nz) cells")
    println("  Radial:")
    println("    Range: $(grid.r_faces[1]) to $(grid.r_faces[end]) m")
    println("    Δr_min = $(grid.dr_min*1000) mm")
    println("    Δr_max = $(grid.dr_max*1000) mm")
    println("    Clustering ratio: $(grid.dr_max/grid.dr_min)")
    println("  Axial:")
    println("    Range: $(grid.z_faces[1]) to $(grid.z_faces[end]) m")
    println("    Δz_min = $(grid.dz_min*1000) mm")
    println("    Δz_max = $(grid.dz_max*1000) mm")
    println("    Clustering ratio: $(grid.dz_max/grid.dz_min)")
end

end # module
