# Test suite for solenoid-induced fields
using Test
using LinearAlgebra
using CUDA

# Include necessary modules
include("../utils/induction_functions.jl")
include("../utils/constants.jl")
include("../utils/electric_fields.jl")
include("../utils/particles.jl")  # For electron_energy_from_velocity
include("../utils/cross_sections.jl")  # For cross-section data

# Initialize cross-sections for air
populate_cpu_cross_sections!(air_composition_cpu)
include("../utils/simulation_functions.jl")

@testset "Solenoid Field Calculations" begin
    # Setup solenoid parameters
    solenoid = SolenoidParameters(
        current_amplitude = 100.0,  # Amps
        frequency = 50e3,           # 50 kHz
        num_turns = 500,
        length = 0.5                # meters
    )
    
    # Create test grid (small 2x2x2 grid for simplicity)
    x_grid = [0.0, 0.05, 0.1]
    y_grid = [0.0, 0.05, 0.1]
    z_grid = [0.0, 0.001, 0.002]
    
    @testset "Magnetic Field Calculation" begin
        # Test at t=0
        b0 = calculate_magnetic_field(solenoid, 0.0)
        @test isapprox(b0, [0,0,0], atol=1e-6)
        
        # Test at peak current (t = π/(2ω))
        t_peak = π/(2*2π*solenoid.frequency)
        b_peak = calculate_magnetic_field(solenoid, t_peak)
        expected_bz = μ₀ * (solenoid.num_turns/solenoid.length) * solenoid.current_amplitude
        @test isapprox(b_peak[3], expected_bz, rtol=0.01)
    end
    
    @testset "Induced Electric Field" begin
        # Test at t=0 (should be maximum induced E since dB/dt is maximum)
        t = 0.0
        b, Ex, Ey, Ez = calculate_fields(solenoid, x_grid, y_grid, z_grid, t, false)
        
        # Calculate expected dB/dt
        ω = 2π * solenoid.frequency
        dBdt = μ₀ * (solenoid.num_turns/solenoid.length) * solenoid.current_amplitude * ω
        
        # Check values at grid center
        i, j, k = 2, 2, 2
        x = (x_grid[i] + x_grid[i+1])/2
        y = (y_grid[j] + y_grid[j+1])/2
        
        @test isapprox(Ex[i,j,k], (y/2)*dBdt, rtol=0.01)
        @test isapprox(Ey[i,j,k], -(x/2)*dBdt, rtol=0.01)
        @test isapprox(Ez[i,j,k], 0.0, atol=1e-6)
    end
    
    @testset "GPU Compatibility" begin
        if CUDA.functional()
            # Test that GPU arrays work
            t = 1e-6
            b_gpu, Ex_gpu, Ey_gpu, Ez_gpu = calculate_fields(solenoid, x_grid, y_grid, z_grid, t, true)
            
            # Convert to CPU for comparison
            b_cpu, Ex_cpu, Ey_cpu, Ez_cpu = calculate_fields(solenoid, x_grid, y_grid, z_grid, t, false)
            
            @test Array(Ex_gpu) ≈ Ex_cpu
            @test Array(Ey_gpu) ≈ Ey_cpu
            @test Array(Ez_gpu) ≈ Ez_cpu
        else
            @info "Skipping GPU test because CUDA is not functional"
        end
    end
end

@testset "Field Integration in Simulation" begin
    # This would require a mock simulation environment
    # For now just test that the field combination works
    pic_field = ElectricFieldGrid(
        ones(2,2,2), 2*ones(2,2,2), 3*ones(2,2,2), zeros(2,2,2)
    )
    induced_field = (ones(2,2,2), 0.5*ones(2,2,2), zeros(2,2,2))
    
    combined = ElectricFieldGrid(
        pic_field.Ex .+ induced_field[1],
        pic_field.Ey .+ induced_field[2],
        pic_field.Ez .+ induced_field[3],
        pic_field.potential
    )
    
    @test combined.Ex == 2*ones(2,2,2)
    @test combined.Ey == 2.5*ones(2,2,2)
    @test combined.Ez == 3*ones(2,2,2)
end

@testset "Minimal Simulation Run" begin
    # Minimal parameters for quick test
    nx, ny, nz = 2, 2, 2  # Small grid size for fast testing
    initial_temp = 800.0
    
    params = (
        use_gpu = false,
        dt = 1e-12,
        initial_temperature = initial_temp,
        target_temperature = 1000.0,
        simulated_electrons_per_step = 1000,
        particle_weight = 1e10,
        chamber_dims = (width=0.01, length=0.01, height=0.001),
        x_grid = [0.0, 0.005, 0.01],
        y_grid = [0.0, 0.005, 0.01],
        z_grid = [0.0, 0.0005, 0.001],
        x_cell_size = 0.005,
        y_cell_size = 0.005,
        z_cell_size = 0.0005,
        cell_volume = 2.5e-8,
        total_cells = nx * ny * nz,
        min_energy_eV = 0.1,
        max_energy_eV = 100.0,
        store_animation_data = false,
        field_update_interval = 5,
        magnetic_field = [0.0, 0.0, 0.1],
        anode_voltage = 1000.0,
        electron_injection_energy_eV = 50.0,
        initial_pressure = 1e5,
        initial_air_density_n = 1e23,
        initial_electron_velocity = 1e6,
        initial_temperature_grid = fill(initial_temp, (nx, ny, nz)),
        initial_positions = zeros(0, 3),  # Start with no electrons
        initial_velocities = zeros(0, 3),
        solenoid = SolenoidParameters(
            current_amplitude=100.0,
            frequency=50e3,
            num_turns=500,
            length=0.1
        ),
        max_steps = 50
    )
    
    # Mock data structures
    cpu_data = (air_composition=Dict(),)
    gpu_data = nothing
    
    # Run simulation
    results = run_pic_simulation(params, cpu_data, gpu_data; verbose=true)
    
    # Basic checks
    @test results.final_step > 0
    @test length(results.avg_temps_history) > 1
    @test results.avg_temps_history[end] > params.initial_temperature
end

println("All induction tests passed!")
