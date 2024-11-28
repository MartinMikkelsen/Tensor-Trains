using CairoMakie
using TensorTrains

f = x -> cos(1 / (x^3+0.01)) + sin(π*x)
cores = 10  
N = 150 

qtt = interpolating_qtt(f, cores, N)
qtt_rank_revealing = lagrange_rank_revealing(f, cores, N)

is_qtt(qtt)
is_qtt(qtt_rank_revealing)

qtt_values = matricize(qtt,cores)
qtt_values_rank_revealing = matricize(qtt_rank_revealing,cores)

x_points = LinRange(0, 1, 2^cores)
original_values = f.(x_points)

lines(x_points, original_values, label="Original Function")
lines!(x_points, qtt_values_rank_revealing, label="QTT, rank rev.", linestyle=:dash, color=:green)
lines!(x_points, qtt_values, label="QTT", linestyle=:dash, color=:red)
current_figure()
