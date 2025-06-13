using Plots, LinearAlgebra

# --- Temperature vs Time Plot ---
function plot_temperature_vs_time(time_points, avg_temps_history_julia, target_temperature)
    p1 = plot(time_points, avg_temps_history_julia,
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
function plot_efficiency_vs_time(time_points, efficiency_history_julia)
    p4 = plot(time_points, efficiency_history_julia,
              label="Step Efficiency",
              xlabel="Time (µs)", ylabel="Efficiency (%)",
              title="Heating Efficiency per Step",
              grid=true)
    return p4
end


# --- Heatmap Functions for Density and Temperature ---
function heatmap_density_slice(x_grid, y_grid, final_density_grid, slice_index)
    p2 = heatmap(x_grid*1000, y_grid*1000, final_density_grid[:, slice_index, :]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Electron Density at Final State (Mid-Z Slice, with B Field)",
                 aspect_ratio=:auto, color=:viridis,
                 colorbar_title="Electron Density (a.u.)")
    return p2
end

function heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, slice_index)
    p3 = heatmap(x_grid*1000, y_grid*1000, final_temperature_grid[:, slice_index, :]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Air Temperature at Final State (Mid-Z Slice, with B Field)",
                 aspect_ratio=:auto, color=:magma,
                 colorbar_title="Temperature (K)")
    return p3
end

function plot_efficiency_vs_lifetime(results_df)
    p = scatter(results_df.AvgLifetime .* 1e9, results_df.FinalEfficiency,
                xlabel="Average Electron Lifetime (ns)",
                ylabel="Final Efficiency (%)",
                title="Efficiency vs Electron Lifetime",
                legend=false,
                marker=(:circle, 6),
                color=:purple,
                grid=true)
    
    if length(results_df.AvgLifetime) > 1
        # Manual linear regression
        x = results_df.AvgLifetime .* 1e9
        y = results_df.FinalEfficiency
        A = [x ones(length(x))]
        coeffs = A \ y  # Least squares solution
        plot!(p, x, A * coeffs,
              line=(:dot, 3, :red),
              label="Linear Trend")
    end
    
    return p
end

# --- Electron Positions Scatterplot ---
function plot_electron_positions(positions, x_lim, z_lim; title="")
    p = scatter(positions[:,1]*1000, positions[:,3]*1000,  # Convert to mm
               xlabel="x (mm)",
               ylabel="z (mm)",
               title=title,
               legend=false,
               markersize=2,
               markercolor=:blue,
               xlim=(0, x_lim*1000),
               ylim=(0, z_lim*1000))
    
    # Draw chamber outline
    plot!(p, [0, x_lim*1000, x_lim*1000, 0, 0],
          [0, 0, z_lim*1000, z_lim*1000, 0],
          color=:black, linewidth=2, linestyle=:solid, label="Chamber")
    return p
end

# --- Electron Positions Animation ---
function animate_electron_positions(position_history, x_lim, z_lim;
                                   filename="plots/electron_positions_animation.gif",
                                   fps=10)
    # Create animation
    anim = @animate for (step, positions) in enumerate(position_history)
        title_str = "Electron Positions (Step $step)"
        plot_electron_positions(positions, x_lim, z_lim; title=title_str)
    end
    
    # Save animation as GIF
    gif(anim, filename, fps=fps)
    println("Electron positions animation saved to: $filename")
    return anim
end

function plot_charge_density_slice(x_grid, y_grid, charge_density_grid, slice_index)
    p = heatmap(x_grid*1000, y_grid*1000, charge_density_grid[:, slice_index, :]',
                xlabel="x (mm)", ylabel="y (mm)",
                title="Charge Density (Z Slice)",
                aspect_ratio=:auto, color=:RdBu,
                colorbar_title="Charge Density (C/m³)")
    return p
end

function plot_potential_xz_slice(x_grid, z_grid, potential_grid, y_slice_index; title="Electric Potential (X-Z Slice)")
    # For 1D potential (constant in x-y), take first y-slice
    slice = potential_grid[:, 1, :]
    v_min = minimum(potential_grid)
    v_max = maximum(potential_grid)
    
    p = heatmap(x_grid*1000, z_grid*1000, slice',
                xlabel="x (mm)", ylabel="z (mm)",
                title=title,
                aspect_ratio=:auto, color=:viridis,
                clims=(v_min, v_max),
                colorbar=false)  # Remove colorbar
    return p
end

function plot_electric_field_vectors(x_grid, z_grid, Ex, Ey, slice_index; step=3, title="")
    # Create electric field vector visualization
    x_plot = x_grid[1:step:end] * 1000  # mm
    z_plot = z_grid[1:step:end] * 1000  # mm
    
    Ex_slice = Ex[1:step:end, slice_index, 1:step:end]'
    Ez_slice = Ey[1:step:end, slice_index, 1:step:end]'
    
    p = quiver(x_plot, z_plot, quiver=(Ex_slice, Ez_slice),
              xlabel="x (mm)", ylabel="z (mm)",
              title=isempty(title) ? "Electric Field Vectors (X-Z Slice)" : title,
              aspect_ratio=:auto)
    return p
end

function animate_electric_field(electric_field_history, x_grid, z_grid, slice_index; step=3, fps=10, filename="electric_field_animation.gif", max_frames=50)
    # Limit frames to prevent OOM
    frame_step = max(1, ceil(Int, length(electric_field_history)/max_frames))
    frame_indices = 1:frame_step:length(electric_field_history)
    anim = @animate for i in frame_indices
        field_grid = electric_field_history[i]
        plot_electric_field_vectors(x_grid, z_grid,
                                   field_grid.Ex,
                                   field_grid.Ez,
                                   slice_index;
                                   step=step,
                                   title="Step $i: Electric Field")
    end
    
    # Save animation as GIF
    gif(anim, filename, fps=fps)
    println("Animation saved to: $filename (frames: $(length(frame_indices)))")
    return anim
end
function animate_potential_slice(potential_history, x_grid, z_grid, slice_index; fps=100, filename="plots/potential_animation.gif", max_frames=50)
    # Ensure we have enough frames to animate
    if length(potential_history) < 2
        @warn "Not enough potential history frames to animate (only $(length(potential_history)) available)"
        return nothing
    end
    
    # Calculate target frames count (minimum 10, maximum max_frames)
    target_frames = min(max(10, length(potential_history)), max_frames)
    
    # Calculate frame step
    frame_step = max(1, ceil(Int, length(potential_history)/target_frames))
    frame_indices = 1:frame_step:length(potential_history)
    
    # Ensure we don't exceed target_frames
    if length(frame_indices) > target_frames
        frame_step = ceil(Int, length(potential_history)/target_frames)
        frame_indices = 1:frame_step:length(potential_history)
    end
    
    anim = @animate for i in frame_indices
        potential_grid = potential_history[i]
        plot_potential_xz_slice(x_grid, z_grid, potential_grid, slice_index;
                             title="Step $i: Electric Potential")
    end
    
    # Save animation as GIF
    gif(anim, filename, fps=fps)
    println("Potential animation saved to: $filename (frames: $(length(frame_indices)))")
    return anim
end
function animate_electron_positions(position_history, x_lim, z_lim;
                                 filename="plots/electron_positions_animation.gif",
                                 fps=100, max_frames=50)
    # Check if there is enough data to animate
    if length(position_history) < 2
        @warn "Not enough position data to generate animation (only $(length(position_history)) frames)"
        return nothing
    end
    
    # Calculate target frames count (minimum 10, maximum max_frames)
    target_frames = min(max(10, length(position_history)), max_frames)
    
    # Calculate frame step
    frame_step = max(1, ceil(Int, length(position_history)/target_frames))
    frame_indices = 1:frame_step:length(position_history)
    
    # Create animation for selected frames
    anim = @animate for step in frame_indices
        positions = position_history[step]
        title_str = "Electron Positions (Step $step)"
        plot_electron_positions(positions, x_lim, z_lim; title=title_str)
    end
    
    # Save animation as GIF
    try
        gif(anim, filename, fps=fps)
        println("Position animation saved to: $filename (frames: $(length(frame_indices)))")
    catch e
        @error "Error saving animation: $e"
    end
    
    return anim
end