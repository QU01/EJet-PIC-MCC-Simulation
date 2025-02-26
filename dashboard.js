// Global variables to store loaded data
let temperatureTimeData;
let efficiencyTimeData;
let detailedData;
let temperatureGridData;
let densityGridData;
let simulationParameters;
let bestParameters;
let parameterSearchResults;
let efficiencyEnergyData;
let efficiencyPressureData;
let efficiencyFieldData;

// Chart objects
let temperatureChart;
let efficiencyChart;
let energyTransferChart;
let electronCountChart;
let efficiencyEnergyChart;
let efficiencyPressureChart;
let efficiencyFieldChart;

// Function to initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Load all data
    loadAllData()
        .then(() => {
            // Log after data loading to inspect all data variables
            console.log("Data loading COMPLETED successfully. Inspecting data variables:");
            console.log("temperatureTimeData:", temperatureTimeData);
            console.log("efficiencyTimeData:", efficiencyTimeData);
            console.log("detailedData:", detailedData);
            console.log("temperatureGridData:", temperatureGridData);
            console.log("densityGridData:", densityGridData);
            console.log("simulationParameters:", simulationParameters);
            console.log("bestParameters:", bestParameters);
            console.log("parameterSearchResults:", parameterSearchResults);
            console.log("efficiencyEnergyData:", efficiencyEnergyData);
            console.log("efficiencyPressureData:", efficiencyPressureData);
            console.log("efficiencyFieldData:", efficiencyFieldData);

            // Hide loading alert and initialize visualizations
            document.getElementById('data-loading').classList.add('d-none');
            initializeDashboard();
        })
        .catch(error => {
            console.error('Error loading data:', error);
            document.getElementById('data-loading').classList.add('d-none');
            document.getElementById('data-error').classList.remove('d-none');
        });
});

// Function to load all data files
async function loadAllData() {
    const dataFiles = [
        { name: 'temperatureTimeData', path: 'simulation_data/temperature_vs_time.csv' },
        { name: 'efficiencyTimeData', path: 'simulation_data/efficiency_vs_time.csv' },
        { name: 'detailedData', path: 'simulation_data/detailed_simulation_data.csv' },
        { name: 'temperatureGridData', path: 'simulation_data/temperature_grid.csv' },
        { name: 'densityGridData', path: 'simulation_data/density_grid.csv' },
        { name: 'simulationParameters', path: 'simulation_data/simulation_parameters.csv' },
        { name: 'bestParameters', path: 'simulation_data/best_parameters.csv' },
        { name: 'parameterSearchResults', path: 'simulation_data/parameter_search_results.csv' },
        { name: 'efficiencyEnergyData', path: 'simulation_data/efficiency_vs_energy.csv' },
        { name: 'efficiencyPressureData', path: 'simulation_data/efficiency_vs_pressure.csv' },
        { name: 'efficiencyFieldData', path: 'simulation_data/efficiency_vs_magnetic_field.csv' }
    ];

    const loadPromises = dataFiles.map(file => loadCSV(file.path));
    const results = await Promise.all(loadPromises);

    // Correctly assign results to global variables
    dataFiles.forEach((file, index) => {
        const data = results[index]; // Get the parsed data for this file

        if (file.name === 'temperatureTimeData') {
            temperatureTimeData = data;
        } else if (file.name === 'efficiencyTimeData') {
            efficiencyTimeData = data;
        } else if (file.name === 'detailedData') {
            detailedData = data;
        } else if (file.name === 'temperatureGridData') {
            temperatureGridData = data;
        } else if (file.name === 'densityGridData') {
            densityGridData = data;
        } else if (file.name === 'simulationParameters') {
            simulationParameters = data;
        } else if (file.name === 'bestParameters') {
            bestParameters = data;
        } else if (file.name === 'parameterSearchResults') {
            parameterSearchResults = data;
        } else if (file.name === 'efficiencyEnergyData') {
            efficiencyEnergyData = data;
        } else if (file.name === 'efficiencyPressureData') {
            efficiencyPressureData = data;
        } else if (file.name === 'efficiencyFieldData') {
            efficiencyFieldData = data;
        }
    });
}

// Function to load CSV file
function loadCSV(url) {
    return new Promise((resolve, reject) => {
        Papa.parse(url, {
            download: true,
            header: true,
            dynamicTyping: true,
            complete: function(results) {
                console.log("Papa Parse COMPLETE:", url, results); // DEBUG: Log when Papa Parse completes successfully
                resolve(results.data);
            },
            error: function(error) {
                console.error("Papa Parse ERROR:", url, error); // DEBUG: Log if Papa Parse encounters an error
                reject(error);
            },
            chunkError: function(error, file) {
                console.error("Papa Parse CHUNK ERROR:", url, error, file); // DEBUG: Log if Papa Parse has chunk errors
            }
        });
    });
}

// Initialize dashboard
function initializeDashboard() {
    // Set optimal parameters
    displayOptimalParameters();

    // Initialize charts
    createTemperatureChart();
    createEfficiencyChart();
    createEnergyTransferChart();
    createElectronCountChart();

    // Initialize parameter comparison charts
    createEfficiencyEnergyChart();
    createEfficiencyPressureChart();
    createEfficiencyFieldChart();

    // Initialize spatial visualizations
    createTemperatureHeatmap();
    createDensityHeatmap();

    // Initialize tables
    populateParametersTable();
    populateSearchResultsTable();
    populateDetailedDataTable();

    // Initialize download buttons
    setupDownloadButtons();
}

// Display optimal parameters
function displayOptimalParameters() {
    console.log("Entering displayOptimalParameters function"); // DEBUG: Log when this function is called
    console.log("bestParameters before find:", bestParameters); // DEBUG: Log the value of bestParameters right before using .find()

    // Find best parameters
    const bestParam = bestParameters.find(param => param.Parameter.includes('Electron Energy'));
    const optEnergy = bestParam ? bestParam.Value : "--";

    const bestPressure = bestParameters.find(param => param.Parameter.includes('Pressure'));
    const optPressure = bestPressure ? bestPressure.Value : "--";

    const bestField = bestParameters.find(param => param.Parameter.includes('Magnetic Field'));
    const optField = bestField ? bestField.Value : "--";

    const bestEfficiency = bestParameters.find(param => param.Parameter.includes('Efficiency'));
    const optEfficiency = bestEfficiency ? bestEfficiency.Value.toFixed(2) : "--";

    // Update DOM
    document.getElementById('opt-energy').textContent = optEnergy;
    document.getElementById('opt-pressure').textContent = optPressure;
    document.getElementById('opt-field').textContent = optField;
    document.getElementById('opt-efficiency').textContent = optEfficiency;

    console.log("Exiting displayOptimalParameters function"); // DEBUG: Log when exiting this function
}

// Create Temperature vs Time Chart
function createTemperatureChart() {
    const ctx = document.getElementById('temperature-chart').getContext('2d');

    // Find target temperature
    const targetTempParam = simulationParameters.find(param => param.Parameter.includes('Target Temperature'));
    const targetTemp = targetTempParam ? targetTempParam.Value : null;

    temperatureChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: temperatureTimeData.map(d => d.Time_microseconds),
            datasets: [
                {
                    label: 'Temperatura (K)',
                    data: temperatureTimeData.map(d => d.Temperature_K),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Temperatura Objetivo',
                    data: Array(temperatureTimeData.length).fill(targetTemp),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Evolución de la Temperatura a lo largo del tiempo'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} K`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Tiempo (μs)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Temperatura (K)'
                    }
                }
            }
        }
    });
}

// Create Efficiency vs Time Chart
function createEfficiencyChart() {
    const ctx = document.getElementById('efficiency-chart').getContext('2d');

    efficiencyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: efficiencyTimeData.map(d => d.Time_microseconds),
            datasets: [{
                label: 'Eficiencia (%)',
                data: efficiencyTimeData.map(d => d.Efficiency_percent),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Eficiencia de la Simulación a lo largo del tiempo'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Eficiencia: ${context.parsed.y.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Tiempo (μs)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Eficiencia (%)'
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// Create Energy Transfer Chart
function createEnergyTransferChart() {
    const ctx = document.getElementById('energy-transfer-chart').getContext('2d');

    energyTransferChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: detailedData.map(d => d.Time_microseconds),
            datasets: [
                {
                    label: 'Energía de Entrada',
                    data: detailedData.map(d => d.Input_Energy_J),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true
                },
                {
                    label: 'Transferencia Total',
                    data: detailedData.map(d => d.Total_Energy_Transfer_J),
                    borderColor: 'rgba(255, 159, 64, 1)',
                    backgroundColor: 'rgba(255, 159, 64, 0.1)',
                    borderWidth: 2,
                    fill: true
                },
                {
                    label: 'Transferencia Inelástica',
                    data: detailedData.map(d => d.Inelastic_Energy_Transfer_J),
                    borderColor: 'rgba(153, 102, 255, 1)',
                    backgroundColor: 'rgba(153, 102, 255, 0.1)',
                    borderWidth: 2,
                    fill: true
                },
                {
                    label: 'Transferencia Elástica',
                    data: detailedData.map(d => d.Elastic_Energy_Transfer_J),
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderWidth: 2,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Transferencia de Energía por Paso'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y.toExponential(2)} J`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Tiempo (μs)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Energía (J)'
                    },
                    type: 'logarithmic'
                }
            }
        }
    });
}

// Create Electron Count Chart
function createElectronCountChart() {
    const ctx = document.getElementById('electron-count-chart').getContext('2d');

    electronCountChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: detailedData.map(d => d.Time_microseconds),
            datasets: [{
                label: 'Número de Electrones',
                data: detailedData.map(d => d.Electron_Count),
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Evolución del Número de Electrones'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Tiempo (μs)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Número de Electrones'
                    }
                }
            }
        }
    });
}

// Create Efficiency vs Energy Chart
function createEfficiencyEnergyChart() {
    const ctx = document.getElementById('efficiency-energy-chart').getContext('2d');

    // Group data by pressure and magnetic field
    const datasets = [];
    const uniquePressures = [...new Set(efficiencyEnergyData.map(d => d.Pressure_MPa))];
    const uniqueFields = [...new Set(efficiencyEnergyData.map(d => d.MagneticField_T))];

    // Generate a color for each combination
    const colors = [
        'rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'
    ];

    let colorIndex = 0;

    uniqueFields.forEach(field => {
        uniquePressures.forEach(pressure => {
            const filteredData = efficiencyEnergyData.filter(
                d => d.Pressure_MPa === pressure && d.MagneticField_T === field
            );

            // Sort by energy
            filteredData.sort((a, b) => a.ElectronEnergy_eV - b.ElectronEnergy_eV);

            if (filteredData.length > 0) {
                datasets.push({
                    label: `P=${pressure} MPa, B=${field} T`,
                    data: filteredData.map(d => ({
                        x: d.ElectronEnergy_eV,
                        y: d.Efficiency_percent
                    })),
                    borderColor: colors[colorIndex % colors.length],
                    backgroundColor: colors[colorIndex % colors.length].replace('1)', '0.1)'),
                    borderWidth: 2,
                    fill: false
                });
                colorIndex++;
            }
        });
    });

    efficiencyEnergyChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Eficiencia vs Energía de Electrones'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Energía de Electrones (eV)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Eficiencia (%)'
                    },
                    min: 0
                }
            }
        }
    });
}

// Create Efficiency vs Pressure Chart
function createEfficiencyPressureChart() {
    const ctx = document.getElementById('efficiency-pressure-chart').getContext('2d');

    // Group data by energy and magnetic field
    const datasets = [];
    const uniqueEnergies = [...new Set(efficiencyPressureData.map(d => d.ElectronEnergy_eV))];
    const uniqueFields = [...new Set(efficiencyPressureData.map(d => d.MagneticField_T))];

    // Generate a color for each combination
    const colors = [
        'rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'
    ];

    let colorIndex = 0;

    uniqueEnergies.forEach(energy => {
        uniqueFields.forEach(field => {
            const filteredData = efficiencyPressureData.filter(
                d => d.ElectronEnergy_eV === energy && d.MagneticField_T === field
            );

            // Sort by pressure
            filteredData.sort((a, b) => a.Pressure_MPa - b.Pressure_MPa);

            if (filteredData.length > 0) {
                datasets.push({
                    label: `E=${energy} eV, B=${field} T`,
                    data: filteredData.map(d => ({
                        x: d.Pressure_MPa,
                        y: d.Efficiency_percent
                    })),
                    borderColor: colors[colorIndex % colors.length],
                    backgroundColor: colors[colorIndex % colors.length].replace('1)', '0.1)'),
                    borderWidth: 2,
                    fill: false
                });
                colorIndex++;
            }
        });
    });

    efficiencyPressureChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Eficiencia vs Presión'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Presión (MPa)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Eficiencia (%)'
                    },
                    min: 0
                }
            }
        }
    });
}

// Create Efficiency vs Magnetic Field Chart
function createEfficiencyFieldChart() {
    const ctx = document.getElementById('efficiency-field-chart').getContext('2d');

    // Group data by energy and pressure
    const datasets = [];
    const uniqueEnergies = [...new Set(efficiencyFieldData.map(d => d.ElectronEnergy_eV))];
    const uniquePressures = [...new Set(efficiencyFieldData.map(d => d.Pressure_MPa))];

    // Generate a color for each combination
    const colors = [
        'rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(255, 206, 86, 1)',
        'rgba(75, 192, 192, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)'
    ];

    let colorIndex = 0;

    uniqueEnergies.forEach(energy => {
        uniquePressures.forEach(pressure => {
            const filteredData = efficiencyFieldData.filter(
                d => d.ElectronEnergy_eV === energy && d.Pressure_MPa === pressure
            );

            // Sort by field
            filteredData.sort((a, b) => a.MagneticField_T - b.MagneticField_T);

            if (filteredData.length > 0) {
                datasets.push({
                    label: `E=${energy} eV, P=${pressure} MPa`,
                    data: filteredData.map(d => ({
                        x: d.MagneticField_T,
                        y: d.Efficiency_percent
                    })),
                    borderColor: colors[colorIndex % colors.length],
                    backgroundColor: colors[colorIndex % colors.length].replace('1)', '0.1)'),
                    borderWidth: 2,
                    fill: false
                });
                colorIndex++;
            }
        });
    });

    efficiencyFieldChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Eficiencia vs Campo Magnético'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'Campo Magnético (T)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Eficiencia (%)'
                    },
                    min: 0
                }
            }
        }
    });
}

// Create Temperature Heatmap
function createTemperatureHeatmap() {
    const container = document.getElementById('temperature-heatmap');

    // Get unique x and y values
    const xValues = [...new Set(temperatureGridData.map(d => d.x_mm))].sort((a, b) => a - b);
    const yValues = [...new Set(temperatureGridData.map(d => d.y_mm))].sort((a, b) => a - b);

    // Calculate width and height
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select('#temperature-heatmap')
        .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', 400)
        .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Create scales
    const x = d3.scaleBand()
        .domain(xValues)
        .range([0, width])
        .padding(0.01);

    const y = d3.scaleBand()
        .domain(yValues)
        .range([height, 0])
        .padding(0.01);

    // Add axes
    svg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % 2 === 0)));

    svg.append('g')
        .call(d3.axisLeft(y).tickValues(y.domain().filter((d, i) => i % 2 === 0)));

    // Temperature values range for color scale
    const minTemp = d3.min(temperatureGridData, d => d.temperature_K);
    const maxTemp = d3.max(temperatureGridData, d => d.temperature_K);

    // Create color scale
    const colorScale = d3.scaleSequential()
        .interpolator(d3.interpolateInferno)
        .domain([minTemp, maxTemp]);

    // Create the heatmap
    svg.selectAll()
        .data(temperatureGridData)
        .enter()
        .append('rect')
            .attr('x', d => x(d.x_mm))
            .attr('y', d => y(d.y_mm))
            .attr('width', x.bandwidth())
            .attr('height', y.bandwidth())
            .style('fill', d => colorScale(d.temperature_K));

    // Add title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .text('Distribución de Temperatura (K)');

    // Add X axis label
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 35)
        .attr('text-anchor', 'middle')
        .text('Posición X (mm)');

    // Add Y axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -35)
        .attr('text-anchor', 'middle')
        .text('Posición Y (mm)');

    // Add a legend
    const legendWidth = width * 0.6;
    const legendHeight = 15;
    const legendX = (width - legendWidth) / 2;
    const legendY = height + 65;

    // Create gradient for legend
    const defs = svg.append('defs');
    const linearGradient = defs.append('linearGradient')
        .attr('id', 'temperature-gradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');

    // Add gradient stops
    linearGradient.selectAll('stop')
        .data(d3.range(0, 1.01, 0.1))
        .enter().append('stop')
            .attr('offset', d => `${d * 100}%`)
            .attr('stop-color', d => colorScale(minTemp + d * (maxTemp - minTemp)));

    // Create legend rectangle
    svg.append('rect')
        .attr('x', legendX)
        .attr('y', legendY)
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#temperature-gradient)');

    // Add legend axis
    const legendScale = d3.scaleLinear()
        .domain([minTemp, maxTemp])
        .range([0, legendWidth]);

    svg.append('g')
        .attr('transform', `translate(${legendX}, ${legendY + legendHeight})`)
        .call(d3.axisBottom(legendScale).ticks(5).tickFormat(d => d.toFixed(0)));

    // Add legend title
    svg.append('text')
        .attr('x', legendX + legendWidth / 2)
        .attr('y', legendY - 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('Temperatura (K)');

    // Create tooltip
    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);

    // Add tooltip event
    svg.selectAll('rect')
        .on('mouseover', function(event, d) {
            if (!d) return; // Skip if data is undefined

            tooltip.transition()
                .duration(200)
                .style('opacity', .9);

            tooltip.html(`X: ${d.x_mm.toFixed(1)} mm<br>Y: ${d.y_mm.toFixed(1)} mm<br>Temperatura: ${d.temperature_K.toFixed(1)} K`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 30) + 'px');
        })
        .on('mouseout', function() {
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        });
}

// Create Density Heatmap
function createDensityHeatmap() {
    const container = document.getElementById('density-heatmap');

    // Get unique x and y values
    const xValues = [...new Set(densityGridData.map(d => d.x_mm))].sort((a, b) => a - b);
    const yValues = [...new Set(densityGridData.map(d => d.y_mm))].sort((a, b) => a - b);

    // Calculate width and height
    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = container.clientWidth - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    // Create SVG
    const svg = d3.select('#density-heatmap')
        .append('svg')
            .attr('width', container.clientWidth)
            .attr('height', 400)
        .append('g')
            .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Create scales
    const x = d3.scaleBand()
        .domain(xValues)
        .range([0, width])
        .padding(0.01);

    const y = d3.scaleBand()
        .domain(yValues)
        .range([height, 0])
        .padding(0.01);

    // Add axes
    svg.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(d3.axisBottom(x).tickValues(x.domain().filter((d, i) => i % 2 === 0)));

    svg.append('g')
        .call(d3.axisLeft(y).tickValues(y.domain().filter((d, i) => i % 2 === 0)));

    // Density values range for color scale
    const minDensity = d3.min(densityGridData, d => d.electron_density);
    const maxDensity = d3.max(densityGridData, d => d.electron_density);

    // Create color scale
    const colorScale = d3.scaleSequential()
        .interpolator(d3.interpolateViridis)
        .domain([minDensity, maxDensity]);

    // Create the heatmap
    svg.selectAll()
        .data(densityGridData)
        .enter()
        .append('rect')
            .attr('x', d => x(d.x_mm))
            .attr('y', d => y(d.y_mm))
            .attr('width', x.bandwidth())
            .attr('height', y.bandwidth())
            .style('fill', d => colorScale(d.electron_density));

    // Add title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px')
        .text('Distribución de Densidad de Electrones');

    // Add X axis label
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height + 35)
        .attr('text-anchor', 'middle')
        .text('Posición X (mm)');

    // Add Y axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -height / 2)
        .attr('y', -35)
        .attr('text-anchor', 'middle')
        .text('Posición Y (mm)');

    // Add a legend
    const legendWidth = width * 0.6;
    const legendHeight = 15;
    const legendX = (width - legendWidth) / 2;
    const legendY = height + 65;

    // Create gradient for legend
    const defs = svg.append('defs');
    const linearGradient = defs.append('linearGradient')
        .attr('id', 'density-gradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '100%')
        .attr('y2', '0%');

    // Add gradient stops
    linearGradient.selectAll('stop')
        .data(d3.range(0, 1.01, 0.1))
        .enter().append('stop')
            .attr('offset', d => `${d * 100}%`)
            .attr('stop-color', d => colorScale(minDensity + d * (maxDensity - minDensity)));

    // Create legend rectangle
    svg.append('rect')
        .attr('x', legendX)
        .attr('y', legendY)
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', 'url(#density-gradient)');

    // Add legend axis
    const legendScale = d3.scaleLinear()
        .domain([minDensity, maxDensity])
        .range([0, legendWidth]);

    svg.append('g')
        .attr('transform', `translate(${legendX}, ${legendY + legendHeight})`)
        .call(d3.axisBottom(legendScale).ticks(5).tickFormat(d => d.toExponential(1)));

    // Add legend title
    svg.append('text')
        .attr('x', legendX + legendWidth / 2)
        .attr('y', legendY - 5)
        .attr('text-anchor', 'middle')
        .style('font-size', '12px')
        .text('Densidad de Electrones');

    // Create tooltip
    const tooltip = d3.select('body').append('div')
        .attr('class', 'tooltip')
        .style('opacity', 0);

    // Add tooltip event
    svg.selectAll('rect')
        .on('mouseover', function(event, d) {
            if (!d) return; // Skip if data is undefined

            tooltip.transition()
                .duration(200)
                .style('opacity', .9);

            tooltip.html(`X: ${d.x_mm.toFixed(1)} mm<br>Y: ${d.y_mm.toFixed(1)} mm<br>Densidad: ${d.electron_density.toExponential(2)}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 30) + 'px');
        })
        .on('mouseout', function() {
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        });
}

// Populate Parameters Table
function populateParametersTable() {
    const tableBody = document.querySelector('#param-table tbody');
    tableBody.innerHTML = '';

    simulationParameters.forEach(param => {
        const row = document.createElement('tr');

        const nameCell = document.createElement('td');
        nameCell.textContent = param.Parameter;
        row.appendChild(nameCell);

        const valueCell = document.createElement('td');
        // Format value based on type
        if (typeof param.Value === 'number') {
            if (param.Value < 0.01 && param.Value !== 0) {
                valueCell.textContent = param.Value.toExponential(2);
            } else {
                valueCell.textContent = param.Value.toFixed(4);
            }
        } else {
            valueCell.textContent = param.Value;
        }
        row.appendChild(valueCell);

        tableBody.appendChild(row);
    });
}

// Populate Search Results Table
function populateSearchResultsTable() {
    const tableBody = document.querySelector('#search-results-table tbody');
    tableBody.innerHTML = '';

    // Sort results by efficiency in descending order
    const sortedResults = [...parameterSearchResults].sort((a, b) => b.Efficiency - a.Efficiency);

    // Show top 10 results
    const topResults = sortedResults.slice(0, 10);

    topResults.forEach((result, index) => {
        const row = document.createElement('tr');

        const rankCell = document.createElement('td');
        rankCell.textContent = index + 1;
        row.appendChild(rankCell);

        const energyCell = document.createElement('td');
        energyCell.textContent = result.ElectronEnergy;
        row.appendChild(energyCell);

        const pressureCell = document.createElement('td');
        pressureCell.textContent = (result.Pressure / 1e6).toFixed(1);
        row.appendChild(pressureCell);

        const fieldCell = document.createElement('td');
        fieldCell.textContent = result.MagneticField;
        row.appendChild(fieldCell);

        const efficiencyCell = document.createElement('td');
        efficiencyCell.textContent = result.Efficiency.toFixed(2);
        row.appendChild(efficiencyCell);

        const tempCell = document.createElement('td');
        tempCell.textContent = result.FinalTemperature.toFixed(0);
        row.appendChild(tempCell);

        tableBody.appendChild(row);
    });
}

// Populate Detailed Data Table
function populateDetailedDataTable() {
    const tableBody = document.querySelector('#detailed-data-table tbody');
    tableBody.innerHTML = '';

    // Combine detailed data with temperature data
    const combinedData = detailedData.map((detail, index) => {
        return {
            ...detail,
            Temperature_K: index < temperatureTimeData.length ? temperatureTimeData[index+1]?.Temperature_K : null
        };
    });

    // Show only every 10th row if too many rows (for performance)
    const dataToShow = combinedData.length > 200
        ? combinedData.filter((_, i) => i % 10 === 0)
        : combinedData;

    dataToShow.forEach(data => {
        const row = document.createElement('tr');

        const timeCell = document.createElement('td');
        timeCell.textContent = data.Time_microseconds?.toFixed(3) || '--';
        row.appendChild(timeCell);

        const electronCell = document.createElement('td');
        electronCell.textContent = data.Electron_Count || '--';
        row.appendChild(electronCell);

        const tempCell = document.createElement('td');
        tempCell.textContent = data.Temperature_K?.toFixed(1) || '--';
        row.appendChild(tempCell);

        const inputEnergyCell = document.createElement('td');
        inputEnergyCell.textContent = data.Input_Energy_J?.toExponential(2) || '--';
        row.appendChild(inputEnergyCell);

        const transferEnergyCell = document.createElement('td');
        transferEnergyCell.textContent = data.Total_Energy_Transfer_J?.toExponential(2) || '--';
        row.appendChild(transferEnergyCell);

        const efficiencyCell = document.createElement('td');
        efficiencyCell.textContent = data.Efficiency_percent?.toFixed(1) || '--';
        row.appendChild(efficiencyCell);

        tableBody.appendChild(row);
    });
}

// Setup Download Buttons
function setupDownloadButtons() {
    // Temperature vs Time download
    document.getElementById('download-temp-time').addEventListener('click', function(e) {
        e.preventDefault();
        downloadCSV('simulation_data/temperature_vs_time.csv', 'temperature_vs_time.csv');
    });

    // Efficiency vs Time download
    document.getElementById('download-efficiency-time').addEventListener('click', function(e) {
        e.preventDefault();
        downloadCSV('simulation_data/efficiency_vs_time.csv', 'efficiency_vs_time.csv');
    });

    // Detailed data download
    document.getElementById('download-detailed').addEventListener('click', function(e) {
        e.preventDefault();
        downloadCSV('simulation_data/detailed_simulation_data.csv', 'detailed_simulation_data.csv');
    });
}

// Helper function to download CSV
function downloadCSV(url, filename) {
    fetch(url)
        .then(response => response.blob())
        .then(blob => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        })
        .catch(error => {
            console.error('Error downloading CSV:', error);
            alert('Error downloading file: ' + error.message);
        });
}