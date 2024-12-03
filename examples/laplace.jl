using CairoMakie
using TensorTrains

function boundary_term(cores::Int64, N::Int64)
    bc_left(y) = sin(π .* y)
    bc_right(y) = cos.(π .* y)
    bc_bottom(x) = sin.(π .* x)
    bc_top(x) = cos.(π .* x)

    boundary_left = interpolating_qtt(bc_left, cores, N)
    boundary_right = interpolating_qtt(bc_right, cores, N)
    boundary_bottom = interpolating_qtt(bc_bottom, cores, N)
    boundary_top = interpolating_qtt(bc_top, cores, N)

    b_qtt = boundary_left 
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

    visualize(A_qtt)
    visualize(b_qtt)

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
