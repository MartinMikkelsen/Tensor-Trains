using Base.Threads
using TensorOperations
import Base.+
import Base.*
import LinearAlgebra.dot

"""
Addition of two TTvector
"""
function +(x::TTvector{T,d},y::TTvector{T,d}) where {T<:Number,d}
    @assert(x.ttv_dims == y.ttv_dims, "Dimensions mismatch!")
    ttv_vec = Array{Array{T,3},1}(undef,d)
    rks = x.ttv_rks + y.ttv_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize ttv_vec
    @threads for k in 1:d
        ttv_vec[k] = zeros(T,x.ttv_dims[k],rks[k],rks[k+1])
    end
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
    return TTvector{T,d}(ttv_vec,x.ttv_dims,rks,zeros(d))
end

"""
Addition of two TToperators
"""
function +(x::TToperator{T,d},y::TToperator{T,d}) where {T<:Number,d}
    @assert(x.tto_dims == y.tto_dims, DimensionMismatch)
    tto_vec = Array{Array{T,4},1}(undef,d)
    rks = x.tto_rks + y.tto_rks
    rks[1] = 1
    rks[d+1] = 1
    #initialize tto_vec
    @threads for k in 1:d
        tto_vec[k] = zeros(T,x.tto_dims[k],x.tto_dims[k],rks[k],rks[k+1])
    end
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
    return TToperator{T,d}(tto_vec,x.tto_dims,rks,zeros(d))
end


#matrix vector multiplication in TT format
function *(A::TToperator{T,d},v::TTvector{T,d}) where {T<:Number,d}
    @assert(A.tto_dims==v.ttv_dims, DimensionMismatch)
    Y = Array{Array{T,3},1}(undef, d)
    A_rks = A.tto_rks #R_0, ..., R_d
    v_rks = v.ttv_rks #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A.tto_dims[k], A_rks[k],v_rks[k], A_rks[k+1],v_rks[k+1])
		@tensor M[a,b,c,d,e] = A.tto_vec[k][a,z,b,d]*v.ttv_vec[k][z,c,e]
        Y[k] = reshape(M, A.tto_dims[k], A_rks[k]*v_rks[k], A_rks[k+1]*v_rks[k+1])
    end
    return TTvector{T,d}(Y,A.tto_dims,A.tto_rks.*v.ttv_rks,zeros(Integer,d))
end

#matrix matrix multiplication in TT format
function *(A::TToperator{T,d},B::TToperator{T,d}) where {T<:Number,d}
    @assert(A.tto_dims==B.tto_dims, DimensionMismatch)
    Y = Array{Array{T,4},1}(undef, d)
    A_rks = A.tto_rks #R_0, ..., R_d
    B_rks = B.tto_rks #r_0, ..., r_d
    @threads for k in 1:d
		M = zeros(A.tto_dims[k],A.tto_dims[k],A_rks[k],B_rks[k],A_rks[k+1],B_rks[k+1])
		@tensor M[a,b,c,d,e,f] = A.tto_vec[k][a,z,c,e]*B.tto_vec[k][z,b,d,f]
        Y[k] = reshape(M, A.tto_dims[k], A.tto_dims[k], A_rks[k]*B_rks[k], A_rks[k+1]*B_rks[k+1])
    end
    return TToperator{T,d}(Y,A.tto_dims,A.tto_rks.*B.tto_rks,zeros(Integer,d))
end


#dot returns the dot product of two TTvector
function dot(A::TTvector{T,d},B::TTvector{T,d}) where {T<:Number,d}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
    A_rks = A.ttv_rks
    B_rks = B.ttv_rks
	out = zeros(T,maximum(A_rks),maximum(B_rks))
    out[1,1] = convert(T,1.0)
    @inbounds for k in 1:d
        M = @view(out[1:A_rks[k+1],1:B_rks[k+1]])
		@tensoropt((α,β), M[a,b] = A.ttv_vec[k][z,α,a]*B.ttv_vec[k][z,β,b]*out[1:A_rks[k],1:B_rks[k]][α,β]) #size R^A_{k} × R^B_{k} 
    end
    return out[1,1]::T
end

"""
`dot_par(x_tt,y_tt)' returns the dot product of `x_tt` and `y_tt` in a parallelized algorithm
"""
function dot_par(A::TTvector{T,d},B::TTvector{T,d}) where {T<:Number,d}
    @assert A.ttv_dims==B.ttv_dims "TT dimensions are not compatible"
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


function *(a::S,A::TTvector{R,d}) where {S<:Number,R<:Number,d}
    i = findfirst(isequal(0),A.ttv_ot)
    T = typejoin(typeof(a),R)
    X = copy(A.ttv_vec)
    X[i] = a*X[i]
    return TTvector{T,d}(X,A.ttv_dims,A.ttv_rks,A.ttv_ot)
end

function *(a::S,A::TToperator{R,d}) where {S<:Number,R<:Number,d}
    i = findfirst(isequal(0),A.tto_ot)
    T = typejoin(typeof(a),R)
    X = copy(A.tto_vec)
    X[i] = a*X[i]
    return TToperator{T,d}(X,A.tto_dims,A.tto_rks,A.tto_ot)
end