using BenchmarkTools
using Plots
using Profile
using SparseArrays
using TensorTrains

include("../src/utils.jl")


function right_hand_side(cores::Int64)::Array{Float64,2}
    points = 2^cores
    a, b, d, e = 0, 1, 0, 1
    x = range(a, stop=b, length=points)
    y = range(d, stop=e, length=points)
    f = ((b - a) / points) * ((e - d) / points) .* source_term(x, y, points)
    bound = boundary_term(x, y, points)
    return reshape(f .- bound, points, points)
end

"""
Discrete Laplacian in d dimensions and n equidistant discretization points in each direction
"""
function Lap(n::Int64, d::Int64)::SparseMatrixCSC{Float64,Int64}
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]

    # Iterate over all elements to set up the discrete Laplacian
    for j in 0:n^d-1
        J = digits(j, base=n, pad=d)  # Convert index to d-tuple

        # Set diagonal value
        push!(rows, j + 1)
        push!(cols, j + 1)
        push!(vals, 2 * d)

        # Interaction with neighbors based on boundary conditions
        for k in 0:d-1
            # Check boundary conditions
            if J[k+1] == 0  # Left boundary of the dimension
                # Only set right neighbor
                push!(rows, j + 1)
                push!(cols, j + 1 + n^k)
                push!(vals, -1)
            elseif J[k+1] == n - 1  # Right boundary of the dimension
                # Only set left neighbor
                push!(rows, j + 1)
                push!(cols, j + 1 - n^k)
                push!(vals, -1)
            else
                # Set both neighbors
                push!(rows, j + 1)
                push!(cols, j + 1 - n^k)
                push!(vals, -1)

                push!(rows, j + 1)
                push!(cols, j + 1 + n^k)
                push!(vals, -1)
            end
        end
    end

    # Create a sparse matrix from the lists of row indices, column indices, and values
    return sparse(rows, cols, vals, n^d, n^d)
end

function source_term(x, y, points)
    X = repeat(x', points, 1)
    Y = repeat(y, 1, points)
    return zeros(points, points)
end

function boundary_term(x, y, points::Int)::Array{Float64,2}
    bc_left(y) = sin.(y)
    bc_right(y) = cos.(y)
    bc_bottom(x) = exp.(x)
    bc_top(x) = sin.(x)

    b_bottom_top = zeros(points, points)
    b_left_right = zeros(points, points)

    # Apply boundary conditions using broadcasting
    b_bottom_top[1, :] .= bc_bottom.(x)         # Bottom boundary
    b_bottom_top[end, :] .= bc_top.(x)          # Top boundary

    b_left_right[:, 1] .= bc_left.(y)           # Left boundary
    b_left_right[:, end] .= bc_right.(y)        # Right boundary

    # Combine the adjustments for boundary conditions
    return b_left_right .+ b_bottom_top
end

function solve_Laplace(cores::Int)::Array{Float64,2}
    reshape_dims = ntuple(_ -> 2, 2 * cores * 2)
    L = Lap(2^cores, 2)
    L_TT = tto_decomp(reshape(Array(L), reshape_dims))
    b_dims = repeat([2], 2 * cores)
    b = reshape(right_hand_side(cores), b_dims...)
    b_tt = ttv_decomp(b, index=1, tol=1e-5)

    x_tt = als_linsolv(L_TT, b_tt, b_tt, sweep_count=4)

    y = ttv_to_tensor(x_tt)
    reshape(y, 2^cores, 2^cores)
end

K = solve_Laplace(7)
plot_solution(7, -K)
