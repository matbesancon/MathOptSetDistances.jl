function _check_dimension(v::AbstractVector, s)
    length(v) != MOI.dimension(s) && throw(DimensionMismatch("Mismatch between value and set"))
    return nothing
end

function distance_to_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.Reals) where {T <: Real}
    _check_dimension(v, s)
    return zero(T)
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, s::MOI.Zeros) where {p}
    _check_dimension(v, s)
    return LinearAlgebra.norm(v, p)
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.Zeros)
    return distance_to_set(NormedEpigraphDistance{2}(), v, s)
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, s::MOI.Nonnegatives) where {p}
    _check_dimension(v, s)
    return LinearAlgebra.norm((ifelse(vi < 0, -vi, zero(vi)) for vi in v), p)
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.Nonnegatives)
    return distance_to_set(NormedEpigraphDistance{2}(), v, s)
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, s::MOI.Nonpositives) where {p}
    _check_dimension(v, s)
    return LinearAlgebra.norm((ifelse(vi > 0, vi, zero(vi)) for vi in v), p)
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.Nonpositives)
    _check_dimension(v, s)
    return distance_to_set(NormedEpigraphDistance{2}(), v, s)
end

distance_to_set(::EpigraphViolationDistance, v::Real, s::MOI.LessThan) = max(v - s.upper, zero(v))
distance_to_set(::EpigraphViolationDistance, v::Real, s::MOI.GreaterThan) = max(s.lower - v, zero(v))
distance_to_set(::EpigraphViolationDistance, v::Real, s::MOI.EqualTo) = abs(v - s.value)

distance_to_set(::DefaultDistance, v, s::Union{MOI.LessThan, MOI.GreaterThan, MOI.EqualTo}) = distance_to_set(EpigraphViolationDistance(), v, s)

distance_to_set(::DefaultDistance, v::T, s::MOI.Interval) where {T <: Real} = max(s.lower - v, v - s.upper, zero(T))

function distance_to_set(::EpigraphViolationDistance, v::AbstractVector{<:Real}, s::MOI.NormInfinityCone)
    _check_dimension(v, s)
    t = v[1]
    xs = v[2:end]
    result = maximum(abs, xs) - t
    return max(result, zero(result))
end

function distance_to_set(::EpigraphViolationDistance, v::AbstractVector{<:Real}, s::MOI.NormOneCone)
    _check_dimension(v, s)
    t = v[1]
    xs = v[2:end]
    result = sum(abs, xs) - t
    return max(result, zero(result))
end

function distance_to_set(::EpigraphViolationDistance, v::AbstractVector{<:Real}, s::MOI.SecondOrderCone)
    _check_dimension(v, s)
    t = v[1]
    xs = v[2:end]
    result = LinearAlgebra.norm2(xs) - t
    return max(result, zero(result))
end

function distance_to_set(::DefaultDistance, v, s::Union{MOI.NormInfinityCone, MOI.NormOneCone, MOI.SecondOrderCone})
    return distance_to_set(EpigraphViolationDistance(), v, s)
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, s::MOI.RotatedSecondOrderCone) where {p}
    _check_dimension(v, s)
    t = v[1]
    u = v[2]
    xs = v[3:end]
    return LinearAlgebra.norm(
        (max(-t, zero(t)), max(-u, zero(u)), max(LinearAlgebra.dot(xs,xs) - 2 * t * u)),
        p,
    )
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.RotatedSecondOrderCone)
    return distance_to_set(NormedEpigraphDistance{2}(), v, s)
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.GeometricMeanCone)
    _check_dimension(v, s)
    t = v[1]
    xs = v[2:end]
    n = MOI.dimension(s) - 1
    xresult = LinearAlgebra.norm2(
        max.(-xs, zero(eltype(xs)))
    )
    # early returning if there exists x[i] < 0 to avoid complex sqrt
    if xresult > 0
        return xresult
    end
    result = t - prod(xs)^(inv(n))
    return max(result, zero(result))
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, s::MOI.ExponentialCone) where {p}
    _check_dimension(v, s)
    x = v[1]
    y = v[2]
    z = v[3]
    result = y * exp(x/y) - z
    return LinearAlgebra.norm(
        (max(-y, zero(result)), max(result, zero(result))),
        p,
    )
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.ExponentialCone)
    return distance_to_set(NormedEpigraphDistance{2}(), v, s)
end

function distance_to_set(::NormedEpigraphDistance{p}, vs::AbstractVector{<:Real}, s::MOI.DualExponentialCone) where {p}
    _check_dimension(vs, s)
    u = vs[1]
    v = vs[2]
    w = vs[3]
    result = -u*exp(v/u) - ℯ * w
    return LinearAlgebra.norm(
        (max(u, zero(result)), max(result, zero(result))),
        p,
    )
end

function distance_to_set(::DefaultDistance, vs::AbstractVector{<:Real}, s::MOI.DualExponentialCone)
    return distance_to_set(NormedEpigraphDistance{2}(), vs, s)
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, s::MOI.PowerCone)
    _check_dimension(v, s)
    x = v[1]
    y = v[2]
    z = v[3]
    e = s.exponent
    # early return to avoid complex exponent results
    if x < 0 || y < 0
       return LinearAlgebra.norm2(
            (max(-x, zero(x)), max(-y, zero(x)))
        )
    end
    result = abs(z) - x^e * y^(1-e)
    return max(result, zero(result))
end

function distance_to_set(::NormedEpigraphDistance{p}, vs::AbstractVector{<:Real}, s::MOI.DualPowerCone) where {p}
    _check_dimension(vs, s)
    u = vs[1]
    v = vs[2]
    w = vs[3]
    e = s.exponent
    ce = 1-e
    result = abs(w) - (u/e)^e * (v/ce)^ce
    return LinearAlgebra.norm(
        (max(-u, zero(result)), max(-v, zero(result)), max(result, zero(result))),
        p,
    )
end

function distance_to_set(::DefaultDistance, vs::AbstractVector{<:Real}, s::MOI.DualPowerCone)
    return distance_to_set(NormedEpigraphDistance{2}(), vs, s)
end

function distance_to_set(::NormedEpigraphDistance{p}, v::AbstractVector{<:Real}, set::MOI.RelativeEntropyCone) where {p}
    _check_dimension(v, s)
    n = (dimension(set)-1) ÷ 2
    u = v[1]
    v = v[2:(n+1)]
    w = v[(n+2):end]
    s = sum(w[i] * log(w[i]/v[i]) for i in eachindex(w))
    result = s - u
    return LinearAlgebra.norm(
        push!(
            max.(v[2:end], zero(result)),
            max(result, zero(result)),
        ),
        p,
    )
end

function distance_to_set(::DefaultDistance, v::AbstractVector{<:Real}, set::MOI.RelativeEntropyCone)
    return distance_to_set(NormedEpigraphDistance{2}(), v, set)
end

function distance_to_set(::EpigraphViolationDistance, v::AbstractVector{<:Real}, s::MOI.NormSpectralCone)
    _check_dimension(v, s)
    t = v[1]
    m = reshape(v[2:end], (s.row_dim, s.column_dim))
    s1 = LinearAlgebra.svd(m).S[1]
    result = s1 - t
    return max(result, zero(result))
end

function distance_to_set(::EpigraphViolationDistance, v::AbstractVector{<:Real}, s::MOI.NormNuclearCone)
    _check_dimension(v, s)
    t = v[1]
    m = reshape(v[2:end], (s.row_dim, s.column_dim))
    s1 = sum(LinearAlgebra.svd(m).S)
    result = s1 - t
    return max(result, zero(result))
end

function distance_to_set(::DefaultDistance, v, s::Union{MOI.NormSpectralCone, MOI.NormNuclearCone})
    return distance_to_set(EpigraphViolationDistance(), v, s)
end

function distance_to_set(::DefaultDistance, v::T, ::MOI.ZeroOne) where {T <: Real}
    return min(abs(v - zero(T)), abs(v - one(T)))
end

function distance_to_set(::DefaultDistance, v::Real, ::MOI.Integer)
    return min(abs(v - floor(v)), abs(v - ceil(v)))
end

# return the element-wise distance to zero, with the greatest element to 0
function distance_to_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.SOS1) where {T <: Real}
    _check_dimension(v, s)
    _, i = findmax(abs.(v))
    return LinearAlgebra.norm2([v[j] for j in eachindex(v) if j != i])
end

# takes in input [z, f(x)]
function distance_to_set(d::DefaultDistance, v::AbstractVector{T}, s::MOI.IndicatorSet{A}) where {A, T <: Real}
    _check_dimension(v, s)
    z = v[1]
    # inactive constraint
    if A === MOI.ACTIVATE_ON_ONE && isapprox(z, 0) || A === MOI.ACTIVATE_ON_ZERO && isapprox(z, 1)
        return zeros(T, 2)
    end
    return LinearAlgebra.norm2(
        (distance_to_set(d, z, MOI.ZeroOne()), distance_to_set(v[2], s.set))
    )
end


# find expression of projections on cones and their derivatives here:
#   https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf


"""
    projection_on_set(::AbstractDistance, ::MOI.Zeros, z::Array{Float64}, dual=true)

projection of vector `z` on zero cone i.e. K = {0} or its dual
"""
function projection_on_set(::DefaultDistance, ::MOI.Zeros, z::Array{Float64}, dual=true)
    return dual ? z : zeros(Float64, size(z))
end

function projection_on_set(::DefaultDistance, ::MOI.EqualTo, z::Array{Float64}, dual=true)
    return dual ? z : zeros(Float64, size(z))
end

function projection_on_set(::DefaultDistance, ::MOI.EqualTo, z::Float64, dual=true)
    return dual ? [z] : zeros(Float64, 1)
end

"""
    projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function projection_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{Float64})
    return max.(z, 0.0)
end

"""
    projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_on_set(::DefaultDistance, ::MOI.SecondOrderCone, z::Array{Float64})
    t = z[1]
    x = z[2:length(z)]
    norm_x = LinearAlgebra.norm(x)
    if norm_x <= t
        return copy(z)
    elseif norm_x <= -t
        return zeros(Float64, size(z))
    else
        result = zeros(Float64, size(z))
        result[1] = 1.0
        result[2:length(z)] = x / norm_x
        result *= (norm_x + t) / 2.0
        return result
    end
end

"""
    projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_on_set(::DefaultDistance, ::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64})
    dim = Int64(floor(√(2*length(z))))
    X = unvec_symm(z, dim)
    λ, U = LinearAlgebra.eigen(X)
    return vec_symm(U * LinearAlgebra.Diagonal(max.(λ,0)) * U')
end

"""
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
            X[j,i] = x[(i-1)*dim-Int(((i-1)*i)/2)+j]
        end
    end
    X = X + X'
    X /= √2
    for i in 1:dim
        X[i,i] /= √2
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
    X = copy(X)
    X *= √2
    for i in 1:size(X)[1]
        X[i,i] /= √2
    end
    return X[LinearAlgebra.tril(trues(size(X)))]
end

"""
    projection_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)

Projection onto R^n x K^* x R_+
 `cones` represents a convex cone K, and K^* is its dual cone
"""
function projection_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)
    @assert length(cones) == length(z)
    return vcat([projection_on_set(DefaultDistance(), cones[i], z[i]) for i in 1:length(cones)]...)
end


#  Derivative of the projection of vector `z` on MOI set `cone`
#  projection_gradient_on_set[i,j] = ∂projection_on_set[i] / ∂z[j]   where `projection_on_set` denotes projection of `z` on `cone`

"""
    derivative of projection of vector `z` on zero cone i.e. K = {0}
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.Zeros, z::Array{Float64})
    y = ones(Float64, size(z))
    return reshape(y, length(y), 1)
end

function projection_gradient_on_set(::DefaultDistance, ::MOI.EqualTo, z::Array{Float64})
    y = ones(Float64, size(z))
    return reshape(y, length(y), 1)
end

function projection_gradient_on_set(::DefaultDistance, ::MOI.EqualTo, ::Float64)
    y = ones(Float64, 1)
    return reshape(y, length(y), 1)
end

"""
    derivative of projection of vector `z` on Nonnegative cone i.e. K = R+
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.Nonnegatives, z::Array{Float64})
    y = (sign.(z) .+ 1.0)/2
    n = length(y)
    result = zeros(n, n)
    result[LinearAlgebra.diagind(result)] .= y
    return result
end

"""
    derivative of projection of vector `z` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }
"""
function projection_gradient_on_set(::DefaultDistance, ::MOI.SecondOrderCone, z::Array{Float64})
    n = length(z)
    t = z[1]
    x = z[2:n]
    norm_x = LinearAlgebra.norm(x)
    if norm_x <= t
        return Matrix{Float64}(LinearAlgebra.I,n,n)
    elseif norm_x <= -t
        return zeros(n,n)
    else
        result = [
            norm_x     x';
            x          (norm_x + t)*Matrix{Float64}(LinearAlgebra.I,n-1,n-1) - (t/(norm_x^2))*(x*x')
        ]
        result /= (2.0 * norm_x)
        return result
    end
end

"""
    derivative of projection of vector `z` on positive semidefinite cone i.e. K = S^n⨥
"""
function projection_gradient_on_set(::DefaultDistance, cone::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64})
    n = length(z)
    y = zeros(n)
    D = zeros(n,n)

    for i in 1:n
        y[i] = 1.0
        D[i, 1:n] = projection_gradient_on_set(cone, z, y)
        y[i] = 0.0
    end

    return D
end

function projection_gradient_on_set(::DefaultDistance, ::MOI.PositiveSemidefiniteConeTriangle, z::Array{Float64}, y::Array{Float64})
    n = length(z)
    dim = Int64(floor(√(2*n)))
    X = unvec_symm(z, dim)
    λ, U = LinearAlgebra.eigen(X)

    # if all the eigenvalues are >= 0
    if max.(λ, 0) == λ
        return Matrix{Float64}(LinearAlgebra.I, n, n)
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
                λpi = max(λ[i], 0.0)
                λmj = -min(λ[j], 0.0)
                B[i, j] *= λpi / (λmj + λpi)
            elseif (i <= k && j > k) 
                λmi = -min(λ[i], 0.0)
                λpj = max(λ[j], 0.0)
                B[i, j] *= λpj / (λmi + λpj)
            end
        end
    end

    return vec_symm(U * B * U')
end

"""
    derivative of projection of vector `z` on a product of cones
"""
function projection_gradient_on_set(::DefaultDistance, cones::Array{<:MOI.AbstractSet}, z)
    @assert length(cones) == length(z)
    return BlockDiagonal([projection_gradient_on_set(DefaultDistance(), cones[i], z[i]) for i in 1:length(cones)])
end
