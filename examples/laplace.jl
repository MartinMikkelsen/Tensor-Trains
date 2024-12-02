using BenchmarkTools
using Plots
using SparseArrays
using TensorTrains

function right_hand_side(cores::Int64)::Array{Float64,2}
    points = 2^cores
    a, b, d, e = 0, 1, 0, 1
    x = range(a, stop=b, length=points)
    y = range(d, stop=e, length=points)
    f = ((b - a) / points) * ((e - d) / points) .* source_term(x, y, points)
    bound = boundary_term(x, y, points)
    return reshape(f .- bound, points, points)
end

function source_term(x, y, points)
    return zeros(points, points)
end

function boundary_term(cores::Int64, N::Int64)::QTToperator{Float64,2}
    
    bc_left(y) = sin.(y)
    bc_right(y) = cos.(y)
    bc_bottom(x) = exp.(x)
    bc_top(x) = sin.(x)

    boundary_left = lagrange_rank_revealing(bc_left, cores, N)
    boundary_right = lagrange_rank_revealing(bc_right, cores, N)
    boundary_top = lagrange_rank_revealing(bc_top, cores, N)
    boundary_bottom = lagrange_rank_revealing(bc_bottom, cores, N)

    ket_1 = χ(cores, 1, 0)
    ket_2 = χ(cores, 0, 1)

    return 
end

function solve_Laplace(cores::Int)::Array{Float64,2}
    reshape_dims = ntuple(_ -> 2, 2 * cores * 2)
    L = Laplace_tensor(2^cores, 2)
    L_TT = tto_decomp(reshape(Array(L), reshape_dims))
    b_dims = repeat([2], 2 * cores)
    b = reshape(right_hand_side(cores), b_dims...)
    b_tt = ttv_decomp(b, index=1, tol=1e-5)

    x_tt = als_linsolv(L_TT, b_tt, b_tt, sweep_count=2)

    y = ttv_to_tensor(x_tt)
    reshape(y, 2^cores, 2^cores)
end

K = solve_Laplace(5)
plot_solution(5, -K) #possible sign error
