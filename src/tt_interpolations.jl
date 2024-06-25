using LinearAlgebra

"""
Generate Chebyshev-Lobatto nodes.

# Arguments
- `N::Int`: Number of nodes.

# Returns
- `Vector{Float64}`: Chebyshev-Lobatto nodes.
"""
chebyshev_lobatto_nodes(N::Int)::Vector{Float64} = (cos.(π * (0:N) ./ N) .+ 1) ./ 2

"""
Evaluate Lagrange basis polynomials at specific points.

# Arguments
- `nodes::Vector{Float64}`: Nodes of the polynomial.
- `x::Float64`: Point at which to evaluate the polynomial.
- `j::Int`: Index of the basis polynomial.

# Returns
- `Float64`: Value of the Lagrange basis polynomial at `x`.
"""
function lagrange_basis(nodes::Vector{Float64}, x::Float64, j::Int)::Float64
    N = length(nodes) - 1
    prod([(x - nodes[k]) / (nodes[j] - nodes[k]) for k in 1:N+1 if k != j])
end

"""
Construct QTT representation of a function.

# Arguments
- `f::Function`: Function to represent.
- `core::Int`: Number of cores in the QTT.
- `N::Int`: Number of Chebyshev nodes per core.

# Returns
- `TTvector{Float64}`: QTT representation of the function.
"""
function interpolating_qtt(f::Function, core::Int, N::Int)::TTvector{Float64}
    nodes = chebyshev_lobatto_nodes(N)
    cores = Vector{Array{Float64, 3}}()

    # First core
    A1 = [f((σ + nodes[β]) / 2) for σ in 0:1, β in 1:N+1]
    push!(cores, reshape(A1, 2, 1, N+1))

    # Intermediate cores
    for d in 2:core-1
        Ak = [lagrange_basis(nodes, (σ + nodes[β]) / 2, α) for σ in 0:1, α in 1:N+1, β in 1:N+1]
        push!(cores, reshape(Ak, 2, N+1, N+1))
    end

    # Last core
    AR = [lagrange_basis(nodes, σ / 2, α) for σ in 0:1, α in 1:N+1]
    push!(cores, reshape(AR, 2, N+1, 1))

    dims = ntuple(_ -> 2, core)
    rks = [1; repeat([N+1], core-1); 1]
    TTvector{Float64, core}(core, cores, dims, rks, zeros(Int, core))
end

"""
Interpolate using the QTT at dyadic points.

# Arguments
- `qtt::TTvector{Float64}`: QTT representation.
- `core::Int`: Number of cores in the QTT.

# Returns
- `Vector{Float64}`: Interpolated values at dyadic points.
"""
function interpolate_qtt_at_dyadic_points(qtt::TTvector{Float64}, core::Int)::Vector{Float64}
    full_tensor = ttv_to_tensor(qtt)
    n = 2^core
    values = zeros(n)

    for i in 1:n
        x_le_p = sum(((i >> (k-1)) & 1) / 2^k for k in 1:core)  # Calculate the dyadic point
        index_bits = bitstring(i - 1)[end-core+1:end]  # Binary representation
        indices = [parse(Int, bit) + 1 for bit in index_bits]  # Indices for CartesianIndex
        values[i] = full_tensor[CartesianIndex(indices...)]
    end

    values
end
