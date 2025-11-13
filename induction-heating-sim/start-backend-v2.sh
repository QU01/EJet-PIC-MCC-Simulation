#!/bin/bash

# Start Backend Server v2 for Induction Heating Simulation
# With refined mesh and CUDA support

echo "========================================"
echo "Starting Julia Backend Server v2"
echo "Features: Refined Mesh + CUDA Support"
echo "========================================"

cd backend

# Check if Julia is installed
if ! command -v julia &> /dev/null; then
    echo "Error: Julia is not installed or not in PATH"
    echo "Please install Julia from https://julialang.org/downloads/"
    exit 1
fi

# Check if Project.toml exists
if [ ! -f "Project.toml" ]; then
    echo "Error: Project.toml not found"
    echo "Are you in the correct directory?"
    exit 1
fi

# Install dependencies if needed
echo "Checking Julia dependencies..."
echo "(This may take a while if CUDA.jl needs to be installed)"
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo ""
echo "Configuration:"
echo "  - Non-uniform mesh with wall refinement"
echo "  - CUDA GPU acceleration (auto-detected)"
echo ""

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "No NVIDIA GPU detected - will use CPU mode"
    echo ""
fi

# Start server v2
echo "Starting server on port 8080..."
echo "Press Ctrl+C to stop"
echo ""

julia --project=. src/server_v2.jl
