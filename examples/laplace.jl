using CairoMakie
using TensorTrains

function delta_tt(cores::Int, position::Int)
    dims = ntuple(_ -> 2, cores)
    rks = [1; fill(1, cores - 1); 1]
    cores_vec = Vector{Array{Float64, 3}}(undef, cores)
    for i in 1:cores
        cores_vec[i] = zeros(Float64, dims[i], rks[i], rks[i+1])
        # Adjust for 1-based indexing
        bit = ((position - 1) >>> (cores - i)) & 1
        cores_vec[i][bit + 1, 1, 1] = 1.0
    end
    return TTvector{Float64, cores}(cores, cores_vec, dims, rks, zeros(Int64, cores))
end

function kron_tt(tt1::TTvector{T}, tt2::TTvector{T}) where {T}
    N1 = tt1.N
    N2 = tt2.N
    N = N1 + N2
    cores = Vector{Array{T,3}}(undef, N)
    dims = (tt1.ttv_dims..., tt2.ttv_dims...)
    rks = vcat(tt1.ttv_rks[1:end-1], tt1.ttv_rks[end] * tt2.ttv_rks[1], tt2.ttv_rks[2:end])
    for i in 1:N1
        cores[i] = tt1.ttv_vec[i]
    end
    for i in 1:N2
        cores[N1 + i] = tt2.ttv_vec[i]
    end
    ot = vcat(tt1.ttv_ot, tt2.ttv_ot)
    return TTvector{T, N}(N, cores, dims, rks, ot)
end


function boundary_term(cores::Int64, N::Int64)
    bc_left(y) = sin(π .* y)
    bc_right(y) = cos.(π .* y)
    bc_bottom(x) = sin.(π .* x)
    bc_top(x) = cos.(π .* x)

    delta_x0 = delta_tt(cores, 1)
    delta_xN = delta_tt(cores, 2^cores)
    delta_y0 = delta_tt(cores, 1)
    delta_yN = delta_tt(cores, 2^cores)

    left_boundary_term = kron_tt(delta_x0, boundary_left)
    right_boundary_term = kron_tt(delta_xN, boundary_right)
    bottom_boundary_term = kron_tt(boundary_bottom, delta_y0)
    top_boundary_term = kron_tt(boundary_top, delta_yN)
    
    left_boundary_term = kron_tt(delta_x0, boundary_left)
    right_boundary_term = kron_tt(delta_xN, boundary_right)
    bottom_boundary_term = kron_tt(boundary_bottom, delta_y0)
    top_boundary_term = kron_tt(boundary_top, delta_yN)

    b_qtt = left_boundary_term + right_boundary_term + top_boundary_term + bottom_boundary_term
    return b_qtt
end

function solve_Laplace(cores::Int)::Array{Float64,2}
    b_qtt = boundary_term(cores, 50) 

    A = Δ_tto(2^cores, 2, Δ_DD)

    n_qtt_cores = cores 
    row_dims = [[2 for _ in 1:n_qtt_cores] for _ in 1:A.N]
    col_dims = [[2 for _ in 1:n_qtt_cores] for _ in 1:A.N]
    
    A_qtt = tt2qtt(A, row_dims, col_dims)
    x_init = rand_tt(b_qtt.ttv_dims, b_qtt.ttv_rks)

    println("A_qtt ranks: ", A_qtt.tto_rks)
    println("b_qtt ranks: ", b_qtt.ttv_rks)

    x_qtt = als_linsolv(A_qtt, b_qtt, x_init)
    
    Q = ttv_to_tensor(x_qtt)

    Q_w = reshape(Q, (2^cores, 2^cores))
    return Q_w
end


cores = 5
K = solve_Laplace(cores)

# Create spatial grid
x = LinRange(0, 1, 2^cores)
y = LinRange(0, 1, 2^cores)

# Create figure with proper aspect ratio
f = Figure(resolution=(400, 400))
ax = Axis(f[1, 1],
          aspect=1,
          title="Solution of Laplace Equation",
          xlabel="x",
          ylabel="y")

# Plot with spatial coordinates
co = contourf!(ax, x, y, K, 
               levels=100,
               colormap=:viridis)

# Add colorbar
Colorbar(f[1, 2], co)

# Set axis limits and ticks
ax.xticks = 0:0.2:1
ax.yticks = 0:0.2:1

# Display
display(f)
