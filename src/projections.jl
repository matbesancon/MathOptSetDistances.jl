# find expression of projections on cones and their derivatives here:
#   https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Zeros) where {T}

projection of vector `v` on zero cone i.e. K = {0}
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Zeros) where {T}
    return FillArrays.Zeros{T}(size(v))
end

"""
    projection_on_set(::AbstractDistance, ::MOI.Reals, v::Array{T}) where {T}

projection of vector `v` on real cone i.e. K = R
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Reals) where {T}
    return v
end

function projection_on_set(::DefaultDistance, v::T, set::MOI.EqualTo) where {T}
    return zero(T) + set.value
end

function projection_on_set(::DefaultDistance, v::T, set::MOI.LessThan) where {T}
    return min(v, MOI.constant(set))
end

function projection_on_set(::DefaultDistance, v::T, set::MOI.GreaterThan) where {T}
    return max(v, MOI.constant(set))
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonnegatives) where {T}

projection of vector `v` on Nonnegative cone i.e. K = R^n+
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonnegatives) where {T}
    return max.(v, zero(T))
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonpositives) where {T}

projection of vector `v` on Nonpositive cone i.e. K = R^n-
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonpositives) where {T}
    return min.(v, zero(T))
end

"""
    projection_on_set(::NormedEpigraphDistance{p}, v::AbstractVector{T}, ::MOI.SecondOrderCone) where {T}

projection of vector `v` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_on_set(::NormedEpigraphDistance{p}, v::AbstractVector{T}, ::MOI.SecondOrderCone) where {p, T}
    t = v[1]
    x = v[2:length(v)]
    norm_x = LinearAlgebra.norm(x, p)
    if norm_x <= t
        return copy(v)
    elseif norm_x <= -t
        return zeros(T, size(v))
    end
    result = zeros(T, size(v))
    result[1] = one(T)
    result[2:length(v)] = x / norm_x
    result *= (norm_x + t) / 2
    return result
end

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.SecondOrderCone) where {T}
    return projection_on_set(NormedEpigraphDistance{2}(), v, cone)
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PositiveSemidefiniteConeTriangle) where {T}

projection of vector `v` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PositiveSemidefiniteConeTriangle) where {T}
    dim = isqrt(2*length(v))
    X = unvec_symm(v, dim)
    λ, U = LinearAlgebra.eigen(X)
    D = LinearAlgebra.Diagonal(max.(λ, 0))
    return vec_symm(U * D * U')
end

"""
    unvec_symm(x, dim)

Returns a dim-by-dim symmetric matrix corresponding to `x`.

`x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric matrix
```
X = [ X11  X12 ... X1k
      X21  X22 ... X2k
      ...
      Xk1  Xk2 ... Xkk ],
```
where `vec(X) = (X11, X12, X22, X13, X23, X33, ..., Xkk)`.

### Note on inner products

Note that the scalar product for the symmetric matrix in its vectorized form is
the sum of the pairwise product of the diagonal entries plus twice the sum of
the pairwise product of the upper diagonal entries; see [p. 634, 1].
Therefore, this transformation breaks inner products:
```
dot(unvec_symm(x, dim), unvec_symm(y, dim)) != dot(x, y).
```

### References

[1] Boyd, S. and Vandenberghe, L.. *Convex optimization*. Cambridge university press, 2004.
"""
function unvec_symm(x, dim=isqrt(2length(x)))
    X = zeros(eltype(x), dim, dim)
    idx = 1
    for i in 1:dim
        for j in 1:i
            X[j,i] = X[i,j] = x[idx]
            idx += 1
        end
    end
    return X
end

"""
    vec_symm(X)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, X12, X22, X13, X23, X33, ..., Xkk)`

### Note on inner products

Note that the scalar product for the symmetric matrix in its vectorized form is
the sum of the pairwise product of the diagonal entries plus twice the sum of
the pairwise product of the upper diagonal entries; see [p. 634, 1].
Therefore, this transformation breaks inner products:
```
dot(vec_symm(X), vec_symm(Y)) != dot(X, Y).
```

### References

[1] Boyd, S. and Vandenberghe, L.. *Convex optimization*. Cambridge university press, 2004.

"""
function vec_symm(X)
    return X[LinearAlgebra.tril(trues(size(X)))']
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet})

Projection onto `sets`, a product of sets
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet}) where {T}
    length(v) == length(sets) || throw(DimensionMismatch("Mismatch between value and set"))
    return reduce(vcat, (projection_on_set(DefaultDistance(), v[i], sets[i]) for i in eachindex(sets)))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Zeros) where {T}

derivative of projection of vector `v` on zero cone i.e. K = {0}^n
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Zeros) where {T}
    return FillArrays.Zeros(length(v), length(v))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Reals) where {T}

derivative of projection of vector `v` on real cone i.e. K = R^n
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Reals) where {T}
    return LinearAlgebra.Diagonal(ones(length(v)))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::T, ::MOI.EqualTo)
"""
function projection_gradient_on_set(::DefaultDistance, ::T, ::MOI.EqualTo) where {T}
    return zero(T)
end

function projection_gradient_on_set(::DefaultDistance, v::T, s::MOI.LessThan) where {T}
    return oneunit(T) * (v <= MOI.constant(s))
end

function projection_gradient_on_set(::DefaultDistance, v::T, s::MOI.GreaterThan) where {T}
    return oneunit(T) * (v >= MOI.constant(s))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonnegatives) where {T}

derivative of projection of vector `v` on Nonnegative cone i.e. K = R^n+
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonnegatives) where {T}
    y = (sign.(v) .+ one(T))/2
    return LinearAlgebra.Diagonal(y)
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonpositives) where {T}

derivative of projection of vector `v` on Nonpositives cone i.e. K = R^n-
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.Nonpositives) where {T}
    y = (-sign.(v) .+ one(T))/2
    return LinearAlgebra.Diagonal(y)
end

"""
    projection_gradient_on_set(::NormedEpigraphDistance{p}, v::AbstractVector{T}, ::MOI.SecondOrderCone) where {T}

derivative of projection of vector `v` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_gradient_on_set(::NormedEpigraphDistance{p}, v::AbstractVector{T}, ::MOI.SecondOrderCone) where {p,T}
    n = length(v)
    t = v[1]
    x = v[2:n]
    norm_x = LinearAlgebra.norm(x, p)
    if norm_x <= t
        return Matrix{T}(LinearAlgebra.I,n,n)
    elseif norm_x <= -t
        return zeros(T, n, n)
    end
    result = [
        norm_x     x';
        x          (norm_x + t)*Matrix{T}(LinearAlgebra.I,n-1,n-1) - (t/(norm_x^2))*(x*x')
    ]
    result /= (2 * norm_x)
    return result
end

function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.SecondOrderCone) where {T}
    return projection_gradient_on_set(NormedEpigraphDistance{2}(), v, cone)
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.PositiveSemidefiniteConeTriangle) where {T}

derivative of projection of vector `v` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PositiveSemidefiniteConeTriangle) where {T}
    n = length(v)
    dim = isqrt(2n)
    X = unvec_symm(v, dim)
    λ, U = LinearAlgebra.eigen(X)
    Tp = promote_type(T, Float64)

    # if all the eigenvalues are >= 0
    if all(λi ≥ zero(λi) for λi in λ)
        return Matrix{Tp}(LinearAlgebra.I, n, n)
    end

    # k is the number of negative eigenvalues in X minus ONE
    k = count(λi < 1e-4 for λi in λ)

    y = zeros(Tp, n)
    D = zeros(Tp, n, n)

    for idx in 1:n
        # set eigenvector
        y[idx] = 1

        # defining matrix B
        X̃ = unvec_symm(y, dim)
        B = U' * X̃ * U

        for i in 1:size(B)[1] # do the hadamard product
            for j in 1:size(B)[2]
                if (i <= k && j <= k)
                    @inbounds B[i, j] = 0
                elseif (i > k && j <= k)
                    λpi = max(λ[i], zero(Tp))
                    λmj = -min(λ[j], zero(Tp))
                    @inbounds B[i, j] *= λpi / (λmj + λpi)
                elseif (i <= k && j > k)
                    λmi = -min(λ[i], zero(Tp))
                    λpj = max(λ[j], zero(Tp))
                    @inbounds B[i, j] *= λpj / (λmi + λpj)
                end
            end
        end
        @inbounds D[idx, :] = vec_symm(U * B * U')
        # reset eigenvector
        @inbounds y[idx] = 0
    end
    return D
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet})

Derivative of the projection of vector `v` on product of `sets`
projection_gradient_on_set[i,j] = ∂projection_on_set[i] / ∂v[j] where `projection_on_set` denotes projection of `v` on `cone`

Find expression of projections on cones and their derivatives here: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet}) where {T}
    length(v) == length(sets) || throw(DimensionMismatch("Mismatch between value and set"))
    return BlockDiagonal([projection_gradient_on_set(DefaultDistance(), v[i], sets[i]) for i in eachindex(sets)])
end
