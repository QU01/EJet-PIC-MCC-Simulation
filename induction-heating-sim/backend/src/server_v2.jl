"""
HTTP Server for Induction Heating Simulation API (v2)
With non-uniform mesh and CUDA support
"""

using HTTP
using JSON3
using Sockets

include("simulation_v2.jl")
using .SimulationV2
using .Physics

# Global state
current_simulation = nothing
is_running = false

"""
CORS middleware
"""
function cors_middleware(handler)
    return function(req::HTTP.Request)
        response = handler(req)
        HTTP.setheader(response, "Access-Control-Allow-Origin" => "*")
        HTTP.setheader(response, "Access-Control-Allow-Methods" => "GET, POST, OPTIONS")
        HTTP.setheader(response, "Access-Control-Allow-Headers" => "Content-Type")
        return response
    end
end

"""
Handle OPTIONS requests (CORS preflight)
"""
function handle_options(req::HTTP.Request)
    return HTTP.Response(200, [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type"
    ])
end

"""
GET /api/status - Get server status with GPU info
"""
function handle_status(req::HTTP.Request)
    # Check CUDA availability
    cuda_available = false
    cuda_device = "None"

    try
        @eval using CUDA
        if CUDA.functional()
            cuda_available = true
            cuda_device = CUDA.name(CUDA.device())
        end
    catch
        # CUDA not available
    end

    status = Dict(
        "server" => "running",
        "simulation_running" => is_running,
        "version" => "2.0.0",
        "features" => Dict(
            "refined_mesh" => true,
            "cuda_support" => true,
            "cuda_available" => cuda_available,
            "cuda_device" => cuda_device
        )
    )

    return HTTP.Response(200, ["Content-Type" => "application/json"],
                        JSON3.write(status))
end

"""
GET /api/defaults - Get default parameters
"""
function handle_defaults(req::HTTP.Request)
    params = create_default_params()

    defaults = Dict(
        "geometry" => Dict(
            "R_in" => params.R_in,
            "R_out" => params.R_out,
            "L_z" => params.L_z
        ),
        "coil" => Dict(
            "N_coil" => params.N_coil,
            "I0" => params.I₀,
            "frequency" => params.freq,
            "R_coil" => params.R_coil,
            "R_b" => params.R_b
        ),
        "fluid" => Dict(
            "sigma" => params.σ_fluid,
            "cp" => params.c_p,
            "cv" => params.c_v,
            "gamma" => params.γ,
            "R_gas" => params.R_gas
        ),
        "inlet" => Dict(
            "pressure" => params.p_in,
            "temperature" => params.T_in,
            "velocity" => params.u_in
        ),
        "mesh" => Dict(
            "radial_stretching" => 2.0,
            "axial_stretching" => 1.0
        )
    )

    return HTTP.Response(200, ["Content-Type" => "application/json"],
                        JSON3.write(defaults))
end

"""
POST /api/simulate - Run simulation with refined mesh and GPU
"""
function handle_simulate(req::HTTP.Request)
    global is_running, current_simulation

    if is_running
        error_msg = Dict("error" => "Simulation already running")
        return HTTP.Response(400, ["Content-Type" => "application/json"],
                           JSON3.write(error_msg))
    end

    try
        # Parse request body
        body = JSON3.read(String(req.body))

        # Extract parameters
        geom = get(body, :geometry, Dict())
        coil_params = get(body, :coil, Dict())
        fluid_params = get(body, :fluid, Dict())
        inlet_params = get(body, :inlet, Dict())
        sim_params = get(body, :simulation, Dict())
        mesh_params = get(body, :mesh, Dict())

        # Create PhysicalParams
        params = PhysicalParams(
            R_in = get(geom, :R_in, 0.02),
            R_out = get(geom, :R_out, 0.025),
            L_z = get(geom, :L_z, 0.5),
            N_coil = get(coil_params, :N_coil, 100),
            I₀ = get(coil_params, :I0, 500.0),
            freq = get(coil_params, :frequency, 50e3),
            R_coil = get(coil_params, :R_coil, 0.5),
            R_b = get(coil_params, :R_b, 0.03),
            z_coil = get(geom, :L_z, 0.5) / 2,
            σ_fluid = get(fluid_params, :sigma, 500.0),
            c_p = get(fluid_params, :cp, 14300.0),
            c_v = get(fluid_params, :cv, 10200.0),
            γ = get(fluid_params, :gamma, 1.41),
            R_gas = get(fluid_params, :R_gas, 4124.0),
            μ = Physics.μ₀,
            p_in = get(inlet_params, :pressure, 101325.0),
            T_in = get(inlet_params, :temperature, 300.0),
            u_in = get(inlet_params, :velocity, 50.0)
        )

        # Create SimulationConfig with mesh parameters
        config = SimulationConfig(
            t_final = get(sim_params, :t_final, 5.0),
            CFL = get(sim_params, :CFL, 0.5),
            save_interval = get(sim_params, :save_interval, 0.1),
            Nr = get(sim_params, :Nr, 30),
            Nz = get(sim_params, :Nz, 100),
            radial_stretching = get(mesh_params, :radial_stretching, 2.0),
            axial_stretching = get(mesh_params, :axial_stretching, 1.0),
            use_gpu = get(sim_params, :use_gpu, true),  # Auto-detect by default
            output_dir = "./data"
        )

        # Run simulation
        is_running = true
        println("\n[API] Starting simulation v2...")
        println("  Mesh refinement: radial=$(config.radial_stretching), axial=$(config.axial_stretching)")
        println("  GPU: $(config.use_gpu)")

        results = run_simulation(params, config)
        current_simulation = results

        is_running = false
        println("[API] Simulation completed")

        # Convert results to dict
        response_data = results_to_dict(results)

        return HTTP.Response(200, ["Content-Type" => "application/json"],
                           JSON3.write(response_data))

    catch e
        is_running = false
        println("[API] Error during simulation: ", e)
        println(stacktrace(catch_backtrace()))

        error_msg = Dict(
            "error" => "Simulation failed",
            "message" => string(e)
        )
        return HTTP.Response(500, ["Content-Type" => "application/json"],
                           JSON3.write(error_msg))
    end
end

"""
GET /api/results - Get latest simulation results
"""
function handle_results(req::HTTP.Request)
    global current_simulation

    if current_simulation === nothing
        error_msg = Dict("error" => "No simulation results available")
        return HTTP.Response(404, ["Content-Type" => "application/json"],
                           JSON3.write(error_msg))
    end

    response_data = results_to_dict(current_simulation)

    return HTTP.Response(200, ["Content-Type" => "application/json"],
                        JSON3.write(response_data))
end

"""
Main router
"""
function router(req::HTTP.Request)
    # Handle OPTIONS for CORS
    if req.method == "OPTIONS"
        return handle_options(req)
    end

    # Route requests
    if req.target == "/api/status" && req.method == "GET"
        return handle_status(req)
    elseif req.target == "/api/defaults" && req.method == "GET"
        return handle_defaults(req)
    elseif req.target == "/api/simulate" && req.method == "POST"
        return handle_simulate(req)
    elseif req.target == "/api/results" && req.method == "GET"
        return handle_results(req)
    else
        error_msg = Dict("error" => "Not found")
        return HTTP.Response(404, ["Content-Type" => "application/json"],
                           JSON3.write(error_msg))
    end
end

"""
Start the HTTP server
"""
function start_server(port::Int=8080)
    println("=" ^ 70)
    println("INDUCTION HEATING SIMULATION SERVER v2")
    println("=" ^ 70)
    println("Features: Refined mesh + CUDA support")
    println()

    # Check CUDA
    try
        @eval using CUDA
        if CUDA.functional()
            println("✓ CUDA GPU detected: ", CUDA.name(CUDA.device()))
            println("  Memory: ", CUDA.available_memory() ÷ 1024^3, " GB available")
        else
            println("⚠ CUDA.jl loaded but GPU not functional - using CPU only")
        end
    catch
        println("ℹ CUDA.jl not installed - CPU mode only")
        println("  To enable GPU: Run `using Pkg; Pkg.add(\"CUDA\")`")
    end

    println()
    println("Server starting on port $port...")
    println()
    println("Available endpoints:")
    println("  GET  /api/status   - Server status (includes GPU info)")
    println("  GET  /api/defaults - Default parameters")
    println("  POST /api/simulate - Run simulation")
    println("  GET  /api/results  - Get latest results")
    println()
    println("Press Ctrl+C to stop the server")
    println("=" ^ 70)
    println()

    # Create server with CORS middleware
    server = HTTP.serve!(cors_middleware(router), Sockets.localhost, port)

    return server
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    port = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 8080
    start_server(port)
end
