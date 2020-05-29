
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
        (max(-t, zero(t)), max(-u, zero(u)), max(LinearAlgebra.dot(xs,xs) - 2 * t * u)), p
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
        (max(-y, zero(result)), max(result, zero(result))), p,
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
        (max(u, zero(result)), max(result, zero(result))), p,
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
        (max(-u, zero(result)), max(-v, zero(result)), max(result, zero(result))), p,
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
        ), p,
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
