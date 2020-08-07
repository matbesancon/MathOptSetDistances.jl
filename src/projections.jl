# find expression of projections on cones and their derivatives here:
#   https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf


"""
    projection_on_set(::AbstractDistance, ::MOI.Zeros, z::Array{T}) where {T}

projection of vector `z` on zero cone i.e. K = {0}
"""
function projection_on_set(::DefaultDistance, ::MOI.Zeros, z::Array{T}) where {T}
    return zeros(T, size(z))
end

"""
    projection_on_set(::AbstractDistance, ::MOI.Reals, z::Array{T}) where {T}

projection of vector `z` on real cone i.e. K = R
"""
function projection_on_set(::DefaultDistance, ::MOI.Reals, z::Array{T}) where {T}
    return z
end

function projection_on_set(::DefaultDistance, set::MOI.EqualTo, z::T) where {T}
    return zeros(T, size(z)) .+ set.value
end

"""
    projection_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{T}) where {T}

projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function projection_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{T}) where {T}
    return max.(z, zero(T))
end

"""
    projection_on_set(::NormedEpigraphDistance{p}, ::MOI.SecondOrderCone, z::Array{T}) where {T}

projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_on_set(::NormedEpigraphDistance{p}, ::MOI.SecondOrderCone, z::Array{T}) where {T} where {p}
    t = z[1]
    x = z[2:length(z)]
    norm_x = LinearAlgebra.norm(x, p)
    if norm_x <= t
        return copy(z)
    elseif norm_x <= -t
        return zeros(T, size(z))
    else
        result = zeros(T, size(z))
        result[1] = 1.0
        result[2:length(z)] = x / norm_x
        result *= (norm_x + t) / 2.0
        return result
    end
end

function projection_on_set(::DefaultDistance, cone::MOI.SecondOrderCone, z::Array{T}) where {T}
    return projection_on_set(NormedEpigraphDistance{2}(), cone, z)
end

"""
    projection_on_set(::DefaultDistance, ::MOI.PositiveSemidefiniteConeTriangle, z::Array{T}) where {T}
    
projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_on_set(::DefaultDistance, ::MOI.PositiveSemidefiniteConeTriangle, z::Array{T}) where {T}
    dim = isqrt(2*length(z))
    X = unvec_symm(z, dim)
    λ, U = LinearAlgebra.eigen(X)
    D = LinearAlgebra.Diagonal(max.(λ, 0))
    return vec_symm(U * D * U')
end

"""
    unvec_symm(x, dim)

Returns a dim-by-dim symmetric matrix corresponding to `x`.

`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
matrix; the correspondence is as in SCS.
X = [ X11 X12 ... X1k
        X21 X22 ... X2k
        ...
        Xk1 Xk2 ... Xkk ],
where
vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
"""
function unvec_symm(x, dim)
    X = zeros(dim, dim)
    for i in 1:dim
        for j in i:dim
            @inbounds X[j,i] = X[i,j] = x[(i-1)*dim-div((i-1)*i, 2)+j]
        end
    end
    for i in 1:dim
        for j in i+1:dim
            X[i, j] /= √2
            X[j, i] /= √2
        end
    end
    return X
end

"""
    vec_symm(X)

Returns a vectorized representation of a symmetric matrix `X`.
Vectorization (including scaling) as per SCS.
`vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)`
"""
function vec_symm(X)
    for i in 1:size(X)[1]
        for j in i+1:size(X)[2]
            X[i, j] *= √2
            X[j, i] *= √2
        end
    end
    return X[LinearAlgebra.tril(trues(size(X)))]
end

"""
    projection_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)

Projection onto `K`, a product of convex cones

Find expression of projections on cones and their derivatives here: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
"""
function projection_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)
    @assert length(z) == length(cones)
    return vcat([projection_on_set(DefaultDistance(), cones[i], z[i]) for i in 1:length(cones)]...)
end



"""
    projection_gradient_on_set(::DefaultDistance, ::MOI.Zeros, z::Array{T}) where {T}

derivative of projection of vector `z` on zero cone i.e. K = {0}
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.Zeros, z::Array{T}) where {T}
    y = zeros(T, size(z))
    return reshape(y, length(y), 1)
end

"""
    projection_gradient_on_set(::DefaultDistance, ::MOI.Reals, z::Array{T}) where {T}

derivative of projection of vector `z` on real cone i.e. K = R
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.Reals, z::Array{T}) where {T}
    y = ones(T, size(z))
    return reshape(y, length(y), 1)
end

"""
    projection_gradient_on_set(::DefaultDistance, ::MOI.EqualTo, z::T)
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.EqualTo, ::T) where {T}
    y = zeros(T, 1)
    return reshape(y, length(y), 1)
end

"""
    projection_gradient_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{T}) where {T}    

derivative of projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{T}) where {T}
    y = (sign.(z) .+ one(T))/2
    n = length(y)
    result = zeros(T, n, n)
    result[LinearAlgebra.diagind(result)] .= y
    return result
end

"""
    projection_gradient_on_set(::NormedEpigraphDistance{p}, ::MOI.SecondOrderCone, z::Array{T}) where {T}

derivative of projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_gradient_on_set(::NormedEpigraphDistance{p}, ::MOI.SecondOrderCone, z::Array{T}) where {T} where {p}
    n = length(z)
    t = z[1]
    x = z[2:n]
    norm_x = LinearAlgebra.norm(x, p)
    if norm_x <= t
        return Matrix{T}(LinearAlgebra.I,n,n)
    elseif norm_x <= -t
        return zeros(T, n,n)
    else
        result = [
            norm_x     x';
            x          (norm_x + t)*Matrix{T}(LinearAlgebra.I,n-1,n-1) - (t/(norm_x^2))*(x*x')
        ]
        result /= (2.0 * norm_x)
        return result
    end
end

function projection_gradient_on_set(::DefaultDistance, cone::MOI.SecondOrderCone, z::Array{T}) where {T}
    return projection_gradient_on_set(NormedEpigraphDistance{2}(), cone, z)
end

"""
    projection_gradient_on_set(::DefaultDistance, cone::MOI.PositiveSemidefiniteConeTriangle, z::Array{T}) where {T}

derivative of projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_gradient_on_set(distance::DefaultDistance, cone::MOI.PositiveSemidefiniteConeTriangle, z::Array{T}) where {T}
    n = length(z)
    y = zeros(T, n)
    D = zeros(T, n, n)

    for i in 1:n
        y[i] = one(T)
        @inbounds D[i, 1:n] = projection_gradient_on_set(distance, cone, z, y)
        y[i] = zero(T)
    end

    return D
end

function projection_gradient_on_set(::DefaultDistance, ::MOI.PositiveSemidefiniteConeTriangle, z::Array{T}, y::Array{T}) where {T}
    n = length(z)
    dim = isqrt(2*n)
    X = unvec_symm(z, dim)
    λ, U = LinearAlgebra.eigen(X)

    # if all the eigenvalues are >= 0
    if max.(λ, 0) == λ
        return Matrix{T}(LinearAlgebra.I, n, n)
    end

    # k is the number of negative eigenvalues in X minus ONE
    k = count(λ .< 1e-4)

    # defining matrix B
    X̃ = unvec_symm(y, dim)
    B = U' * X̃ * U
    
    for i in 1:size(B)[1] # do the hadamard product
        for j in 1:size(B)[2]
            if (i <= k && j <= k)
                B[i, j] = 0
            elseif (i > k && j <= k)
                λpi = max(λ[i], zero(T))
                λmj = -min(λ[j], zero(T))
                B[i, j] *= λpi / (λmj + λpi)
            elseif (i <= k && j > k) 
                λmi = -min(λ[i], zero(T))
                λpj = max(λ[j], zero(T))
                B[i, j] *= λpj / (λmi + λpj)
            end
        end
    end

    return vec_symm(U * B * U')
end

"""
    projection_gradient_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)
 
Derivative of the projection of vector `z` on product of `cones`
projection_gradient_on_set[i,j] = ∂projection_on_set[i] / ∂z[j]   where `projection_on_set` denotes projection of `z` on `cone`

Find expression of projections on cones and their derivatives here: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
"""
function projection_gradient_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)
    @assert length(z) == length(cones)
    return BlockDiagonal([projection_gradient_on_set(DefaultDistance(), cones[i], z[i]) for i in 1:length(cones)])
end
