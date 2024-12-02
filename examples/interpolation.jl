using CairoMakie

f = x -> cos(1 / (x^3+0.01)) 

cores = 10  
N = 150 

qtt = interpolating_qtt(f, cores, N)
qtt_rank_revealing = lagrange_rank_revealing(f, cores, N)
visualize(qtt_rank_revealing)
is_qtt(qtt)
is_qtt(qtt_rank_revealing)

qtt_values = matricize(qtt, cores)
qtt_values_rank_revealing = matricize(qtt_rank_revealing, cores)

x_points = LinRange(0, 1, 2^cores)
original_values = f.(x_points)

fig = Figure()
ax = Axis(fig[1, 1], title="Function Interpolation", xlabel="x", ylabel="f(x)")

lines!(ax, x_points, original_values, label="Original Function")
lines!(ax, x_points, qtt_values_rank_revealing, label="QTT, rank rev.", linestyle=:dash, color=:green)
lines!(ax, x_points, qtt_values, label="QTT", linestyle=:dash, color=:red)

axislegend(ax)
fig