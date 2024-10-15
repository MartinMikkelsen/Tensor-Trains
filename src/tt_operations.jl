using Base.Threads
using TensorOperations
import Base.+
import Base.-
import Base.*
import Base./
import LinearAlgebra.dot

"""
Addition of two TTvector
"""
function +(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = x.N
    ttv_vec = Array{Array{T,3},1}(undef,d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize ttv_vec
    @threads for k in 1:d
        ttv_vec[k] = zeros(T,x.ttv_dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        ttv_vec[1][:,:,1:x.ttv_rks[2]] = x.ttv_vec[1]
        ttv_vec[1][:,:,(x.ttv_rks[2]+1):rks[2]] = y.ttv_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            ttv_vec[k][:,1:x.ttv_rks[k],1:x.ttv_rks[k+1]] = x.ttv_vec[k]
            ttv_vec[k][:,(x.ttv_rks[k]+1):rks[k],(x.ttv_rks[k+1]+1):rks[k+1]] = y.ttv_vec[k]
        end
        #last core
        ttv_vec[d][:,1:x.ttv_rks[d],1] = x.ttv_vec[d]
        ttv_vec[d][:,(x.ttv_rks[d]+1):rks[d],1] = y.ttv_vec[d]
        end
    return TTvector{T,N}(d,ttv_vec,x.ttv_dims,rks,zeros(Int64,d))
end

"""
Addition of two TToperators
"""
function +(x::TToperator{T,N},y::TToperator{T,N}) where {T<:Number,N}
    @assert x.tto_dims == y.tto_dims "Incompatible dimensions"
    d = x.N
    tto_vec = Array{Array{T,4},1}(undef,d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize tto_vec
    @threads for k in 1:d
        tto_vec[k] = zeros(T,x.tto_dims[k],x.tto_dims[k],rks[k],rks[k+1])
    end
    @inbounds begin
        #first core 
        tto_vec[1][:,:,:,1:x.tto_rks[1+1]] = x.tto_vec[1]
        tto_vec[1][:,:,:,(x.tto_rks[2]+1):rks[2]] = y.tto_vec[1]
        #2nd to end-1 cores
        @threads for k in 2:(d-1)
            tto_vec[k][:,:,1:x.tto_rks[k],1:x.tto_rks[k+1]] = x.tto_vec[k]
            tto_vec[k][:,:,(x.tto_rks[k]+1):rks[k],(x.tto_rks[k+1]+1):rks[k+1]] = y.tto_vec[k]
        end
        #last core
        tto_vec[d][:,:,1:x.tto_rks[d],1] = x.tto_vec[d]
        tto_vec[d][:,:,(x.tto_rks[d]+1):rks[d],1] = y.tto_vec[d]
    end
    return TToperator{T,N}(d,tto_vec,x.tto_dims,rks,zeros(Int64,d))
end


#matrix vector multiplication in TT format
function *(A::TToperator{T,N},v::TTvector{T,N}) where {T<:Number,N}
    @assert A.tto_dims==v.ttv_dims "Incompatible dimensions"
    y = zeros_tt(T,A.tto_dims,A.tto_rks.*v.ttv_rks)
    @inbounds begin @simd for k in 1:v.N
        yvec_temp = reshape(y.ttv_vec[k], (y.ttv_dims[k], A.tto_rks[k], v.ttv_rks[k], A.tto_rks[k+1], v.ttv_rks[k+1]))
        @tensoropt((νₖ₋₁,νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.tto_vec[k][iₖ,jₖ,αₖ₋₁,αₖ]*v.ttv_vec[k][jₖ,νₖ₋₁,νₖ])
    end end
    return y
end

#matrix matrix multiplication in TT format
function *(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    @assert A.tto_dims==B.tto_dims "Incompatible dimensions"
    d = A.N
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    Y = [zeros(T,A.tto_dims[k], A.tto_dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1]) for k in eachindex(A.tto_dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], A.tto_dims[k], A.tto_dims[k], A_rks[k],B_rks[k], A_rks[k+1],B_rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = A.tto_vec[k][iₖ,z,αₖ₋₁,αₖ]*B.tto_vec[k][z,jₖ,βₖ₋₁,βₖ]
            end
        end
    end
    return TToperator{T,N}(d,Y,A.tto_dims,A.tto_rks.*B.tto_rks,zeros(Int64,d))
end

*(A::TToperator{T,N},B...) where {T,N} = *(A,*(B...))

function *(A::Array{TTvector{T,N},1},x::Vector{T}) where {T,N}
    out = x[1]*A[1]
    for i in 2:length(A)
        out = out + x[i]*A[i]
    end
    return out
end

#dot returns the dot product of two TTvector
function dot(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	out = zeros(T,maximum(A_rks),maximum(B_rks))
    out[1,1] = one(T)
    @inbounds for k in eachindex(A.ttv_dims)
        M = @view(out[1:A_rks[k+1],1:B_rks[k+1]])
		@tensor M[a,b] = A.ttv_vec[k][z,α,a]*(B.ttv_vec[k][z,β,b]*out[1:A_rks[k],1:B_rks[k]][α,β]) #size R^A_{k} × R^B_{k} 
    end
    return out[1,1]::T
end

"""
`dot_par(x_tt,y_tt)' returns the dot product of `x_tt` and `y_tt` in a parallelized algorithm
"""
function dot_par(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
    d = length(A.ttv_dims)
    Y = Array{Array{T,2},1}(undef,d)
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	C = zeros(T,maximum(A_rks.*B_rks))
    @threads for k in 1:d
		M = zeros(T,A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d] = A.ttv_vec[k][z,a,c]*B.ttv_vec[k][z,b,d] #size R^A_{k-1} ×  R^B_{k-1} × R^A_{k} × R^B_{k} 
		Y[k] = reshape(M, A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    @inbounds C[1:length(Y[d])] = Y[d][:]
    for k in d-1:-1:1
        @inbounds C[1:size(Y[k],1)] = Y[k]*C[1:size(Y[k],2)]
    end
    return C[1]::T
end

function *(a::S,A::TTvector{R,N}) where {S<:Number,R<:Number,N}
    T = typejoin(typeof(a),R)
    if iszero(a)
        return zeros_tt(T,A.ttv_dims,ones(Int64,A.N+1))
    else
        i = findfirst(isequal(0),A.ttv_ot)
        X = copy(A.ttv_vec)
        X[i] = a*X[i]
        return TTvector{T,N}(A.N,X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
    end
end

function *(a::S,A::TToperator{R,N}) where {S<:Number,R<:Number,N}
    i = findfirst(isequal(0),A.tto_ot)
    T = typejoin(typeof(a),R)
    X = copy(A.tto_vec)
    X[i] = a*X[i]
    return TToperator{T,N}(A.N,X,A.tto_dims,A.tto_rks,A.tto_ot)
end

function -(A::TTvector{T,N},B::TTvector{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function -(A::TToperator{T,N},B::TToperator{T,N}) where {T<:Number,N}
    return *(-1.0,B)+A
end

function /(A::TTvector,a)
    return 1/a*A
end

"""
returns the matrix x y' in the TTO format
"""
function outer_product(x::TTvector{T,N},y::TTvector{T,N}) where {T<:Number,N}
    Y = [zeros(T,x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k]*y.ttv_rks[k], x.ttv_rks[k+1]*y.ttv_rks[k+1]) for k in eachindex(x.ttv_dims)]
    @inbounds @simd for k in eachindex(Y)
		M_temp = reshape(Y[k], x.ttv_dims[k], x.ttv_dims[k], x.ttv_rks[k], y.ttv_rks[k], x.ttv_rks[k+1],y.ttv_rks[k+1])
        @simd for jₖ in size(M_temp,2)
            @simd for iₖ in size(M_temp,1)
                @tensor M_temp[iₖ,jₖ,αₖ₋₁,βₖ₋₁,αₖ,βₖ] = x.ttv_vec[k][iₖ,αₖ₋₁,αₖ]*conj(y.ttv_vec[k][jₖ,βₖ₋₁,βₖ])
            end
        end
    end
    return TToperator{T,N}(x.N,Y,x.ttv_dims,x.ttv_rks.*y.ttv_rks,zeros(Int64,x.N))
end

"""
Hadamard (element-wise) product of two TTvectors
"""
function ⊙(x::TTvector{T,N}, y::TTvector{T,N}) where {T<:Number,N}
    @assert x.ttv_dims == y.ttv_dims "Incompatible dimensions"
    d = x.N
    ttv_vec = Array{Array{T,3},1}(undef,d)
    rks = x.ttv_rks .* y.ttv_rks

    @inbounds @threads for k in 1:d
        ttv_vec[k] = zeros(T, x.ttv_dims[k], rks[k], rks[k+1])
        for i in 1:x.ttv_dims[k]
            for α in 1:x.ttv_rks[k]
                for β in 1:x.ttv_rks[k+1]
                    for γ in 1:y.ttv_rks[k]
                        for δ in 1:y.ttv_rks[k+1]
                            ttv_vec[k][i, (α-1)*y.ttv_rks[k] + γ, (β-1)*y.ttv_rks[k+1] + δ] =
                                x.ttv_vec[k][i, α, β] * y.ttv_vec[k][i, γ, δ]
                        end
                    end
                end
            end
        end
    end
    return TTvector{T,N}(d, ttv_vec, x.ttv_dims, rks, zeros(Int64,d))
end

function +(x::QTTvector{T,N}, y::QTTvector{T,N}) where {T<:Number,N}
    @assert x.qtt_dims == y.qtt_dims "Incompatible dimensions"
    d = x.N
    qtt_vec = Array{Array{T,3},1}(undef,d)
    rks = x.qtt_rks + y.qtt_rks
    rks[1] = 1
    rks[d+1] = 1
    
    @threads for k in 1:d
        qtt_vec[k] = zeros(T, x.qtt_dims[k], rks[k], rks[k+1])
    end
    
    @inbounds begin
        # First core
        qtt_vec[1][:,:,1:x.qtt_rks[2]] = x.qtt_vec[1]
        qtt_vec[1][:,:,(x.qtt_rks[2]+1):rks[2]] = y.qtt_vec[1]
        
        # Middle cores
        @threads for k in 2:(d-1)
            qtt_vec[k][:,1:x.qtt_rks[k],1:x.qtt_rks[k+1]] = x.qtt_vec[k]
            qtt_vec[k][:,(x.qtt_rks[k]+1):rks[k],(x.qtt_rks[k+1]+1):rks[k+1]] = y.qtt_vec[k]
        end
        
        # Last core
        qtt_vec[d][:,1:x.qtt_rks[d],1] = x.qtt_vec[d]
        qtt_vec[d][:,(x.qtt_rks[d]+1):rks[d],1] = y.qtt_vec[d]
    end
    
    return QTTvector{T,N}(d, qtt_vec, x.qtt_dims, rks, zeros(Int64, d))
end

function *(A::QTToperator{T,N}, v::QTTvector{T,N}) where {T<:Number,N}
    @assert A.qtt_dims == v.qtt_dims "Incompatible dimensions"
    y = zeros_qtt(T, A.qtt_dims, A.qtt_rks .* v.qtt_rks)
    
    @inbounds begin
        @simd for k in 1:v.N
            yvec_temp = reshape(y.qtt_vec[k], (y.qtt_dims[k], A.qtt_rks[k], v.qtt_rks[k], A.qtt_rks[k+1], v.qtt_rks[k+1]))
            @tensoropt((νₖ₋₁, νₖ), yvec_temp[iₖ,αₖ₋₁,νₖ₋₁,αₖ,νₖ] = A.qtt_op_vec[k][iₖ,jₖ,αₖ₋₁,αₖ] * v.qtt_vec[k][jₖ,νₖ₋₁,νₖ])
        end
    end
    
    return y
end

function dot(A::QTTvector{T,N}, B::QTTvector{T,N}) where {T<:Number,N}
    @assert A.qtt_dims == B.qtt_dims "QTT dimensions are not compatible"
    A_rks = A.qtt_rks
    B_rks = B.qtt_rks
    out = zeros(T, maximum(A_rks), maximum(B_rks))
    out[1,1] = one(T)
    
    @inbounds for k in eachindex(A.qtt_dims)
        M = @view(out[1:A_rks[k+1], 1:B_rks[k+1]])
        @tensor M[a,b] = A.qtt_vec[k][z,α,a] * (B.qtt_vec[k][z,β,b] * out[1:A_rks[k],1:B_rks[k]][α,β])
    end
    
    return out[1,1]::T
end
