#!/bin/bash

# Start Backend Server for Induction Heating Simulation

echo "========================================"
echo "Starting Julia Backend Server..."
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
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Start server
echo ""
echo "Starting server on port 8080..."
echo "Press Ctrl+C to stop"
echo ""

julia --project=. src/server.jl
