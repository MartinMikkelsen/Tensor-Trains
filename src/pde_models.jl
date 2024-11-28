using LinearAlgebra

"""
1D discrete gradient matrix
"""
function ∇_matrix(n)
  G = zeros(Float64, n, n)
  for i in 1:n-1
    G[i, i] = 1.0
    G[i, i+1] = -1.0
  end
  G[n, n] = 1.0
  return G
end

"""
n^d discrete gradient in TTO format with rank 2 of
G = g ⊗ id ⊗ … ⊗ id + ⋯ + id ⊗ … ⊗ id ⊗ g
"""
function ∇_tto(n, d; g=[∇_matrix(n) for i in 1:d])
  G_vec = Vector{Array{Float64,4}}(undef, d)
  rks = vcat(1, 2ones(Int64, d-1), 1)
  
  # first TTO core
  G_vec[1] = zeros(n, n, 1, 2)
  G_vec[1][:,:,1,1] = g[1]
  G_vec[1][:,:,1,2] = Matrix{Float64}(I, n, n)
  
  for i in 2:d-1
    G_vec[i] = zeros(n, n, 2, 2)
    G_vec[i][:,:,1,1] = Matrix{Float64}(I, n, n)
    G_vec[i][:,:,2,1] = g[i]
    G_vec[i][:,:,2,2] = Matrix{Float64}(I, n, n)
  end
  
  G_vec[d] = zeros(n, n, 2, 1)
  G_vec[d][:,:,1,1] = Matrix{Float64}(I, n, n)
  G_vec[d][:,:,2,1] = g[d]
  
  return TToperator{Float64, d}(d, G_vec, Tuple(n*ones(Int64, d)), rks, zeros(Int64, d))
end

"""
1D discrete shift matrix
"""
function shift_matrix(n)
  S = zeros(Float64, n, n)
  for i in 1:n-1
    S[i, i+1] = 1.0
  end
  return S
end


"""
1d-discrete Laplacian
"""
function Δ_matrix(n)
  return Matrix(SymTridiagonal(2ones(n),-ones(n-1)))
end 

"""
n^d discrete Laplacian in TTO format with rank 2 of
H = h ⊗ id ⊗ … ⊗ id + ⋯ + id ⊗ ⋯ ⊗ id ⊗ h
"""
function Δ_tto(n,d;h=[Δ_matrix(n) for i in 1:d])
  H_vec = Vector{Array{Float64,4}}(undef,d)
  rks = vcat(1,2ones(Int64,d-1),1)
  # first TTO core
  H_vec[1] = zeros(n,n,1,2)
  H_vec[1][:,:,1,1] = h[1]
  H_vec[1][:,:,1,2] = Matrix{Float64}(I,n,n)
  for i in 2:d-1
    H_vec[i] = zeros(n,n,2,2)
    H_vec[i][:,:,1,1] = Matrix{Float64}(I,n,n)
    H_vec[i][:,:,2,1] = h[i]
    H_vec[i][:,:,2,2] = Matrix{Float64}(I,n,n)
  end
  H_vec[d] = zeros(n,n,2,1)
  H_vec[d][:,:,1,1] = Matrix{Float64}(I,n,n)
  H_vec[d][:,:,2,1] = h[d]
  return TToperator{Float64,d}(d,H_vec,Tuple(n*ones(Int64,d)),rks,zeros(Int64,d))
end

"""
  H = -Δ + ∑ₖ₌₁ʳ sₖ|ϕₖ⟩⟨φₖ|
"""
function perturbed_Δ_tto(n,d;hermitian=true,r=1,rks=ones(Int64,d+1))
  H = Δ_tto(n,d)
  s = randn(r)
  for k in 1:r
    ϕₖ = rand_tt(H.tto_dims,rks)
    ϕₖ = 1/norm(ϕₖ)*ϕₖ
    if hermitian 
      H = H+s[k]*outer_product(ϕₖ,ϕₖ)
    else 
      φₖ = rand_tt(H.tto_dims,rks)
      φₖ = 1/norm(φₖ)*φₖ
      H = H+s[k]*outer_product(ϕₖ,φₖ)
    end
  end
  return H
end