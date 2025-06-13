# ---------------------------------------------------------------------------
# ARCHIVO: plotting_functions.jl (Funciones para generar gráficos y animaciones)
# ---------------------------------------------------------------------------

using Plots
using LinearAlgebra
# GPU: Importamos CUDA para poder verificar el tipo de los arrays (isa(..., CuArray))
using CUDA

#pyplot()

# --- Función de utilidad para asegurar que los datos estén en la CPU ---
# Esta función será nuestro principal mecanismo de seguridad.
# Si el dato ya es un Array de CPU, no hace nada.
# Si es un CuArray de GPU, lo copia a la CPU.
to_cpu(data::AbstractArray) = Array(data)
to_cpu(data) = data # Para otros tipos de datos que no son arrays

# --- Temperature vs Time Plot ---
function plot_temperature_vs_time(time_points, avg_temps_history, target_temperature)
    # No se necesita `to_cpu` aquí, ya que los historiales de escalares
    # (como la temperatura promedio) siempre se mantienen en la CPU.
    p1 = plot(time_points, avg_temps_history,
              label="Average Air Temperature",
              xlabel="Time (µs)", ylabel="Temperature (K)",
              title="PIC-MCC Simulation: Air Heating to Target Temperature",
              grid=true)
    hline!(p1, [target_temperature],
           color="red", linestyle=:dash,
           label="Target Temperature ($(target_temperature) K)")
    return p1
end

# --- Efficiency vs Time Plot ---
function plot_efficiency_vs_time(time_points, efficiency_history)
    # Tampoco se necesita `to_cpu` aquí.
    p4 = plot(time_points, efficiency_history,
              label="Step Efficiency",
              xlabel="Time (µs)", ylabel="Efficiency (%)",
              title="Heating Efficiency per Step",
              grid=true)
    return p4
end

# --- Heatmap Functions for Density and Temperature ---
function heatmap_density_slice(x_grid, y_grid, final_density_grid, slice_index)
    # GPU: Aseguramos que la malla de densidad esté en la CPU antes de plotear.
    density_cpu = to_cpu(final_density_grid)
    
    p2 = heatmap(x_grid*1000, y_grid*1000, density_cpu[:, slice_index, :]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Electron Density at Final State (Mid-Z Slice)",
                 aspect_ratio=:auto, color=:viridis,
                 colorbar_title="Electron Density (a.u.)")
    return p2
end

function heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, slice_index)
    # GPU: Aseguramos que la malla de temperatura esté en la CPU.
    temp_cpu = to_cpu(final_temperature_grid)
    
    p3 = heatmap(x_grid*1000, y_grid*1000, temp_cpu[:, slice_index, :]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Air Temperature at Final State (Mid-Z Slice)",
                 aspect_ratio=:auto, color=:magma,
                 colorbar_title="Temperature (K)")
    return p3
end

function plot_efficiency_vs_lifetime(results_df)
    # No se necesita `to_cpu` aquí, ya que DataFrames siempre está en la CPU.
    p = scatter(results_df.AvgLifetime .* 1e9, results_df.FinalEfficiency,
                xlabel="Average Electron Lifetime (ns)",
                ylabel="Final Efficiency (%)",
                title="Efficiency vs Electron Lifetime",
                legend=false, marker=(:circle, 6), color=:purple, grid=true)
    
    if length(results_df.AvgLifetime) > 1
        x = results_df.AvgLifetime .* 1e9
        y = results_df.FinalEfficiency
        A = [x ones(length(x))]
        coeffs = A \ y
        plot!(p, x, A * coeffs, line=(:dot, 3, :red), label="Linear Trend")
    end
    return p
end

# --- Electron Positions Scatterplot ---
function plot_electron_positions(positions, x_lim, z_lim; title="")
    # GPU: Aseguramos que el array de posiciones esté en la CPU.
    pos_cpu = to_cpu(positions)
    
    # Evitar error si no hay partículas que plotear
    if isempty(pos_cpu)
        p = plot(xlabel="x (mm)", ylabel="z (mm)", title=title,
                 xlim=(0, x_lim*1000), ylim=(0, z_lim*1000))
    else
        p = scatter(pos_cpu[:,1]*1000, pos_cpu[:,3]*1000,
                   xlabel="x (mm)", ylabel="z (mm)", title=title,
                   legend=false, markersize=2, markercolor=:blue,
                   xlim=(0, x_lim*1000), ylim=(0, z_lim*1000))
    end
    
    plot!(p, [0, x_lim*1000, x_lim*1000, 0, 0],
          [0, 0, z_lim*1000, z_lim*1000, 0],
          color=:black, linewidth=2, linestyle=:solid, label="Chamber")
    return p
end

# --- Animaciones ---
# Las funciones de animación llaman a las funciones de ploteo individuales,
# que ya se encargan de mover los datos a la CPU. Por lo tanto, no necesitan
# cambios directos, pero se benefician de la robustez añadida.

function animate_electron_positions(position_history, x_lim, z_lim;
                                   filename="plots/electron_positions_animation.gif",
                                   fps=10, max_frames=50)
    if length(position_history) < 2
        @warn "No hay suficientes datos de historial de posiciones para animar."
        return
    end
    
    frame_step = max(1, ceil(Int, length(position_history) / max_frames))
    frame_indices = 1:frame_step:length(position_history)
    
    anim = @animate for i in frame_indices
        # La función plot_electron_positions ya maneja la conversión a CPU.
        plot_electron_positions(position_history[i], x_lim, z_lim; title="Electron Positions (Step $i)")
    end
    
    gif(anim, filename, fps=fps)
    println("Animación de posiciones guardada en: $filename")
end

function plot_charge_density_slice(x_grid, y_grid, charge_density_grid, slice_index)
    # GPU: Aseguramos que la malla de densidad de carga esté en la CPU.
    charge_density_cpu = to_cpu(charge_density_grid)
    
    p = heatmap(x_grid*1000, y_grid*1000, charge_density_cpu[:, slice_index, :]',
                xlabel="x (mm)", ylabel="y (mm)",
                title="Charge Density (Z Slice)",
                aspect_ratio=:auto, color=:RdBu,
                colorbar_title="Charge Density (C/m³)")
    return p
end

function plot_potential_xz_slice(x_grid, z_grid, potential_grid, y_slice_index; title="Electric Potential (X-Z Slice)")
    # GPU: Aseguramos que la malla de potencial esté en la CPU.
    potential_cpu = to_cpu(potential_grid)
    
    slice = potential_cpu[:, y_slice_index, :]
    v_min, v_max = extrema(potential_cpu)
    
    p = heatmap(x_grid*1000, z_grid*1000, slice',
                xlabel="x (mm)", ylabel="z (mm)", title=title,
                aspect_ratio=:auto, color=:viridis, clims=(v_min, v_max))
    return p
end

function plot_electric_field_vectors(x_grid, z_grid, Ex, Ey, Ez, slice_index; step=3, title="")
    # GPU: Aseguramos que los componentes del campo estén en la CPU.
    Ex_cpu, Ey_cpu = to_cpu(Ex), to_cpu(Ey)
    
    x_plot = x_grid[1:step:end] * 1000
    z_plot = z_grid[1:step:end] * 1000
    
    Ex_slice = Ex_cpu[1:step:end, slice_index, 1:step:end]'
    Ez_slice = Ey_cpu[1:step:end, slice_index, 1:step:end]' # Nota: Originalmente usabas Ey para el componente z del plot
    
    p = quiver(x_plot, z_plot, quiver=(Ex_slice, Ez_slice),
              xlabel="x (mm)", ylabel="z (mm)",
              title=isempty(title) ? "Electric Field Vectors (X-Z Slice)" : title,
              aspect_ratio=:auto)
    return p
end

function animate_potential_slice(potential_history, x_grid, z_grid, slice_index;
                                 fps=10, filename="plots/potential_animation.gif", max_frames=50)
    if length(potential_history) < 2
        @warn "No hay suficientes datos de historial de potencial para animar."
        return
    end
    
    frame_step = max(1, ceil(Int, length(potential_history) / max_frames))
    frame_indices = 1:frame_step:length(potential_history)
    
    anim = @animate for i in frame_indices
        # plot_potential_xz_slice ya maneja la conversión a CPU.
        plot_potential_xz_slice(x_grid, z_grid, potential_history[i], slice_index;
                                title="Electric Potential (Step $i)")
    end
    
    gif(anim, filename, fps=fps)
    println("Animación de potencial guardada en: $filename")
end