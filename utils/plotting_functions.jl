using Plots

# --- Función para Graficar Temperatura vs Tiempo ---
function plot_temperature_vs_time(time_points, avg_temps_history_julia, target_temperature)
    p1 = plot(time_points, avg_temps_history_julia,
              label="Temperatura Promedio del Aire",
              xlabel="Tiempo (µs)", ylabel="Temperatura (K)",
              title="Simulación PIC-MCC: Calentamiento del Aire hasta Temperatura Objetivo",
              grid=true)
    hline!(p1, [target_temperature],
           color="red", linestyle=:dash,
           label="Temperatura Objetivo ($(target_temperature) K)")
    return p1
end

# --- Función para Graficar Eficiencia vs Tiempo ---
function plot_efficiency_vs_time(time_points, efficiency_history_julia)
    p4 = plot(time_points, efficiency_history_julia,
              label="Eficiencia por Paso",
              xlabel="Tiempo (µs)", ylabel="Eficiencia (%)",
              title="Eficiencia de Calentamiento por Paso",
              grid=true)
    return p4
end


# --- Funciones Heatmap para Densidad y Temperatura ---
function heatmap_density_slice(x_grid, y_grid, final_density_grid, z_slice_index)
    p2 = heatmap(x_grid*1000, y_grid*1000, final_density_grid[:, :, z_slice_index]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Densidad de Electrones al Final (Slice en z=mitad, con Campo B)",
                 aspect_ratio=:auto, color=:viridis,
                 colorbar_title="Densidad de Electrones (u.a.)")
    return p2
end

function heatmap_temperature_slice(x_grid, y_grid, final_temperature_grid, z_slice_index)
    p3 = heatmap(x_grid*1000, y_grid*1000, final_temperature_grid[:, :, z_slice_index]',
                 xlabel="x (mm)", ylabel="y (mm)",
                 title="Temperatura del Aire al Final (Slice en z=mitad, con Campo B)",
                 aspect_ratio=:auto, color=:magma,
                 colorbar_title="Temperatura (K)")
    return p3
end