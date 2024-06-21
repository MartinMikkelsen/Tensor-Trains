using LinearAlgebra
using BenchmarkTools
using Plots
using SparseArrays
using Distributed
using TensorOperations

include("../src/utils.jl")

function right_hand_side(cores::Int)::Array{Float64,2}
    points = 2^cores
    a, b, d, e = 0, 1, 0, 1
    x = range(a, stop=b, length=points)
    y = range(d, stop=e, length=points)
    f = ((b-a) / points) * ((e-d) / points) .* source_term(x, y, points)
    bound = boundary_term(x, y, points)
    reshape(f .- bound, points, points)
end

function source_term(x, y, points)::Array{Float64,2}
    X = repeat(x', points, 1)
    Y = repeat(y, 1, points)
    π^2 * (1 .+ 4 .* Y .^ 2) .* sin.(π .* X) .* sin.(π .* Y .^ 2) + 2 * π * sin.(π .* X) .* cos.(π .* Y .^ 2)
end

function boundary_term(x, y, points)::Array{Float64,2}

    bc_left(y) = 2*exp(y)
    bc_right(y) = sin(y)^2
    bc_bottom(x) = exp(x)
    bc_top(x) = sin(x)
    
    b = zeros(points, points)
    b[1, :] .= bc_bottom.(x)
    b[end, :] .= bc_top.(x)
    b[:, 1] .= bc_left.(y)
    b[:, end] .= bc_right.(y)
    b
end

function solve_Poisson(cores::Int)::Array{Float64,2}
    reshape_dims = ntuple(_ -> 2, 2 * cores * 2)
    L = Laplace_tensor(2^cores, 2)
    L_TT = tto_decomp(reshape(Array(L), reshape_dims))
    b_dims = repeat([2], 2 * cores)
    b = reshape(right_hand_side(cores), b_dims...)
    b_tt = ttv_decomp(b, index=1, tol=1e-3)

    x_tt = als_linsolv(L_TT, b_tt, b_tt, sweep_count=1)

    y = ttv_to_tensor(x_tt)
    reshape(y, 2^cores, 2^cores)
end

K = solve_Poisson(7)

plot_solution(7,K)