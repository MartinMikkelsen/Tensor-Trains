using TensorTrains

f = x -> sin(x/3)+exp(-5*x)*cos(100*x)^4
core = 8 
N = 350 

qtt = interpolating_qtt(f, core, N)

qtt_values = interpolate_qtt_at_dyadic_points(qtt, core)

x_points = LinRange(0, 1, 2^core)
original_values = f.(x_points)

# Plot both the original function and the reconstructed tensor
plot(x_points, original_values, label="Original Function", lw=2)
plot!(x_points, qtt_values, label="QTT Reconstruction", lw=2, linestyle=:dash, color=:red)