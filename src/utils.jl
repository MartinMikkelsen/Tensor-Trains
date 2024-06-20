using Plots 
using LinearAlgebra
using BenchmarkTools
using TensorTrains
using SparseArrays
using TensorOperations
using Base.Threads

function plot_solution(cores,solve_equation;title="Solution on [0,1] Grid")
    solution = solve_equation

    # Calculate the number of points and generate linearly spaced coordinates
    total_points = 2^cores
    x = LinRange(0, 1, total_points)
    y = LinRange(0, 1, total_points)

    # Plot the solution using a heatmap where x and y coordinates specify the grid
    heatmap_plot = heatmap(x, y, solution,
                           cbar=true,
                           xlabel="x",
                           ylabel="y",
                           title=title)

    # Display the plot
    display(heatmap_plot)
end

function benchmark(core_range;title="Benchmark Times for solve_Poisson Function")

	benchmark_times = []
	for cores in core_range
		result = @benchmark solve_Liouville($cores)
		push!(benchmark_times, mean(result).time / 1e6)
	end

	plot(core_range, benchmark_times, label="Execution Time", marker=:circle,
		xlabel="Cores", ylabel="Time (ms)",
		title=title)
end

function solve_bond_dimensions(cores)
    reshape_dims = ntuple(_ -> 2, 2 * cores * 2)
    L_TT = tto_decomp(reshape(Array(Lap_sp(2^cores, 2)), reshape_dims))
    b_dims = repeat([2], 2 * cores)
    b = reshape(rhs(cores), b_dims...)
    b_tt = ttv_decomp(b)
    x_tt = als_linsolv(L_TT, b_tt, b_tt, sweep_count=1)

    return x_tt
end

function plot_bond_dimensions(tt_vector::TTvector)
    # Extract bond dimensions (TT ranks) from the tensor train vector
    bond_dims = tt_vector.ttv_rks
    
    # Plot the bond dimensions, skipping the first and last as they are always 1 (boundary conditions of the tensor train)
    plot(bond_dims[2:end-1], title="Bond Dimensions for TTvector", xlabel="Cores", ylabel="Bond Dimension", legend=false, marker=:circle,xlims=(0,14))
end

function plot_memory_usage(tt_vector::TTvector)
    element_size = sizeof(eltype(tt_vector.ttv_vec[1]))
    memory_usage = [prod(size(core)) * element_size for core in tt_vector.ttv_vec]
    plot(memory_usage, title="Estimated Memory Usage of TT Cores", xlabel="Cores", ylabel="Memory (bytes)", marker=:circle)
end