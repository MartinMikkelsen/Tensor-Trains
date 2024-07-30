include("sptensors.jl")

using Test

@testset "sptensors" begin
    n = 25
    d = 2
    L = Lap(n, d)
    x_o = ones(size(L,1))
    x_o_tt = ttv_decomp(reshape(x_o, n * ones(Int,d)...), d)
    b = L * x_o
    x_s = map(round, 10 * randn(size(L,1)))
    x_s_tt = ttv_decomp(reshape(x_s, n * ones(Int, d)...),d)
    b_tt = ttv_decomp(reshape(b,n*ones(Int,d)...), d)
    L_spm = sparsetensor_mat(n*ones(Int, d), n*ones(Int, d), L)
    L_tt = tto_spdecomp(L_spm, d)
    x_als_tt = als(L_tt, b_tt, x_s_tt, x_o_tt.ttv_rks)
end