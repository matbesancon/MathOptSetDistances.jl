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
    struct SecondOrderConeRotation end

Linear transformation that is a symmetric involution representing the rotation
between the `MOI.SecondOrderCone` and the `MOI.RotatedSecondOrderCone`.
"""
struct SecondOrderConeRotation end

function _rotate(t::T, u::T) where {T}
    s = inv(sqrt(T(2)))
    return s * (t + u), s * (t - u)
end

function Base.:*(::SecondOrderConeRotation, v::AbstractVector{T}) where {T}
    r = copy(v)
    r[1], r[2] = _rotate(r[1], r[2])
    return r
end

function Base.:*(::SecondOrderConeRotation, M::AbstractMatrix{T}) where {T}
    R = copy(M)
    for col in axes(R, 2)
        R[1, col], R[2, col] = _rotate(R[1, col], R[2, col])
    end
    return R
end

function Base.:*(M::AbstractMatrix{T}, ::SecondOrderConeRotation) where {T}
    R = copy(M)
    for row in axes(R, 1)
        R[row, 1], R[row, 2] = _rotate(R[row, 1], R[row, 2])
    end
    return R
end

function projection_on_set(d::NormedEpigraphDistance, v::AbstractVector{T}, ::MOI.RotatedSecondOrderCone) where {T}
    n = length(v)
    R = SecondOrderConeRotation()
    return R * projection_on_set(d, R * v, MOI.SecondOrderCone(n))
end

function projection_on_set(::DefaultDistance, v::AbstractVector, cone::MOI.RotatedSecondOrderCone)
    return projection_on_set(NormedEpigraphDistance{2}(), v, cone)
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PositiveSemidefiniteConeTriangle) where {T}

Projection of vector `v` on positive semidefinite cone i.e. `K = S^n⨥`
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, set::MOI.PositiveSemidefiniteConeTriangle) where {T}
    X = reshape_vector(v, set)
    λ, U = LinearAlgebra.eigen(X)
    D = LinearAlgebra.Diagonal(max.(λ, 0))
    return vectorize(LinearAlgebra.Symmetric(U * D * U'))
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, set::MOI.Scaled) where {T}

Projection of vector `v` on the scaled version of `set.set`.
"""
function projection_on_set(d::DefaultDistance, v::AbstractVector{T}, set::MOI.Scaled) where {T}
    scale = MOI.Utilities.SetDotScalingVector{T}(set.set)
    D = LinearAlgebra.Diagonal(scale)
    return D * projection_on_set(d, D \ v, set.set)
end

"""
    reshape_vector(x, set::MOI.AbstractSymmetricMatrixSetTriangle)

Returns a `dim`-by-`dim` symmetric matrix corresponding to `x` where
`dim` is `MOI.side_dimension(set)`

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
dot(reshape_vector(x, dim), reshape_vector(y, dim)) != dot(x, y).
```

### References

[1] Boyd, S. and Vandenberghe, L.. *Convex optimization*. Cambridge university press, 2004.
"""
function reshape_vector(x, set::MOI.AbstractSymmetricMatrixSetTriangle)
    dim = MOI.side_dimension(set)
    X = zeros(eltype(x), dim, dim)
    idx = 1
    for i in 1:dim
        for j in 1:i
            X[j,i] = X[i,j] = x[idx]
            idx += 1
        end
    end
    return LinearAlgebra.Symmetric(X)
end

"""
    vectorize(X::LinearAlgebra.Symmetric)

Returns a vectorized representation of a symmetric matrix `X`.
`vec(X) = (X11, X12, X22, X13, X23, X33, ..., Xkk)`

### Note on inner products

Note that the scalar product for the symmetric matrix in its vectorized form is
the sum of the pairwise product of the diagonal entries plus twice the sum of
the pairwise product of the upper diagonal entries; see [p. 634, 1].
Therefore, this transformation breaks inner products:
```julia
dot(vectorize(X), vectorize(Y)) != dot(X, Y).
```

### References

[1] Boyd, S. and Vandenberghe, L.. *Convex optimization*. Cambridge university press, 2004.

"""
function vectorize(X::LinearAlgebra.Symmetric)
    return parent(X)[LinearAlgebra.triu(trues(size(X)))]
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.ExponentialCone) where {T}

projection of vector `v` on closure of the exponential cone
i.e. `cl(Kexp) = {(x,y,z) | y e^(x/y) <= z, y>0 } U {(x,y,z)| x <= 0, y = 0, z >= 0}`.

References:
* [Proximal Algorithms, 6.3.4](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)
by Neal Parikh and Stephen Boyd.
* [Projection, presolve in MOSEK: exponential, and power cones]
(https://docs.mosek.com/slides/2018/ismp2018/ismp-friberg.pdf) by Henrik Friberg
* [Projection onto the exponential cone: a univariate root-finding problem]
(https://docs.mosek.com/whitepapers/expcone-proj.pdf) by Henrik Friberg 2021
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.ExponentialCone; tol=1e-8) where {T}
    _check_dimension(v, s)

    if _in_exp_cone(v; dual=false)
        return SVector{3}(v)
    end
    if _in_exp_cone(-v; dual=true)
        # if in polar cone Ko = -K*
        return zeros(SVector{3,T})
    end
    if v[1] <= 0 && v[2] <= 0
        return @SVector([v[1], 0, max(v[3],0)])
    end

    return _exp_cone_proj_case_4(v; tol=tol)
end

function _in_exp_cone(v::AbstractVector{T}; dual=false, tol=1e-8) where {T}
    if dual
        return (
            (isapprox(v[1], 0, atol=tol) && v[2] >= 0 && v[3] >= 0) ||
            (v[1] < 0 && v[1]*exp(v[2]/v[1]) + ℯ * v[3] >= tol)
        )
    else
        return (
            (v[1] <= 0 && isapprox(v[2], 0, atol=tol) && v[3] >= 0) ||
            (v[2] > 0 && v[2] * exp(v[1] / v[2]) - v[3] <= tol)
        )
    end
end

function _exp_cone_heuristic_projection_grad(v::AbstractVector{T}; tol=1e-8) where {T}
    # Try Heuristic solutions [Friberg 2021, Lemma 5.1]
    # (careful with ordering of cone)
    # vp = proj onto primal cone, vd = proj onto polar cone
    vp = SVector{3,T}(min(v[1], 0), zero(T), max(v[3], 0))
    vd = SVector{3,T}(zero(T), min(v[2], 0), min(v[3], 0))
    mat_1 = true
    if v[2] > 0
        zp = max(v[3], v[2]*exp(v[1]/v[2]))
        if zp - v[3] < norm(vp - v)
            vp = SVector{3,T}(v[1], v[2], zp)
            mat_1 = false
        end
    end
    mat = if mat_1
        @SMatrix(T[
            (v[1] <= 0) 0 0
            0           0 0
            0           0 (v[3] >= 0)
        ])
    elseif vp[3] == v[3]
        SMatrix{3,3,T}(I)
    else
        # (v[1], v[2], v[2]*exp(v[1]/v[2]))
        @SMatrix(T[
            1               0                            0
            0               1                            0
            exp(v[1]/v[2])  exp(v[1]/v[2])*(1-v[1]/v[2]) 0
        ])
    end
    if v[1] > 0
        zd = min(v[3], -v[1]*exp(v[2]/v[1] - 1))
        if v[3] - zd < norm(vd - v)
            vd = SVector{3,T}(v[1], v[2], zd)
        end
    end

    # Check if heuristics above approximately satisfy the optimality conditions
    # Friberg 2021 eq (6)
    opt_norm = norm(vp + vd - v)
    opt_ortho = abs(dot(vp, vd))
    if norm(v - vp) < tol || norm(v - vd) < tol || (opt_norm < tol && opt_ortho < tol)
        return true, mat
    end
    return false, mat
end

function _exp_cone_proj_case_4(v::AbstractVector{T}; tol=1e-8) where {T}
    # Try Heuristic solutions [Friberg 2021, Lemma 5.1]
    # (careful with ordering of cone)
    # vp = proj onto primal cone, vd = proj onto polar cone
    vp = SVector{3,T}(min(v[1], 0), zero(T), max(v[3], 0))
    vd = SVector{3,T}(zero(T), min(v[2], 0), min(v[3], 0))
    if v[2] > 0
        zp = max(v[3], v[2]*exp(v[1]/v[2]))
        if zp - v[3] < norm(vp - v)
            vp = SVector{3,T}(v[1], v[2], zp)
        end
    end
    if v[1] > 0
        zd = min(v[3], -v[1]*exp(v[2]/v[1] - 1))
        if v[3] - zd < norm(vd - v)
            vd = SVector{3,T}(v[1], v[2], zd)
        end
    end

    # Check if heuristics above approximately satisfy the optimality conditions
    # Friberg 2021 eq (6)
    opt_norm = norm(vp + vd - v)
    opt_ortho = abs(dot(vp, vd))
    if norm(v - vp) < tol || norm(v - vd) < tol || (opt_norm < tol && opt_ortho < tol)
        return vp
    end

    # Failure of heuristics -> non heuristic solution
    # Ref: https://docs.mosek.com/slides/2018/ismp2018/ismp-friberg.pdf, p47-48
    # Thm: h(x) is smooth, strictly increasing, and changes sign on domain
    r, s, t = v[1], v[2], v[3]
    h(x) = (((x-1)*r + s) * exp(x) - (r - x*s)*exp(-x))/(x^2 - x + 1) - t

    # Note: won't both be Inf by case 3 of projection
    lb = r > 0 ? 1 - s/r : -Inf
    ub = s > 0 ? r/s : Inf

    # Deal with ±Inf bounds
    if isinf(lb)
        lb = min(ub-0.125, -0.125)
        for _ in 1:10
            h(lb) < 0 && break
            ub = lb
            lb *= 2
        end
    end
    if isinf(ub)
        ub = max(lb+0.125, 0.125)
        for _ in 1:10
            h(ub) > 0 && break
            lb = ub
            ub *= 2
        end
    end

    # Check bounds
    if !(h(lb) < 0 && h(ub) > 0)
        error("Failure to find bracketing interval for exp cone projection.")
    end

    x = _bisection(h, lb, ub)
    if x === nothing
        error("Failure in root-finding for exp cone projection with boundaries ($lb, $ub).")
    end

    return ((x - 1) * r + s)/(x^2 - x + 1) * SVector{3,T}(x, 1, exp(x))
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.DualExponentialCone) where {T}

projection of vector `v` on the dual exponential cone
i.e. `Kexp^* = {(u,v,w) | u < 0, -u*exp(v/u) <= ew } U {(u,v,w)| u == 0, v >= 0, w >= 0}`.

References:
* [Proximal Algorithms, 6.3.4](https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf)
by Neal Parikh and Stephen Boyd.
* [Projection, presolve in MOSEK: exponential, and power cones](https://docs.mosek.com/slides/2018/ismp2018/ismp-friberg.pdf)
by Henrik Friberg
"""
function projection_on_set(d::DefaultDistance, v::AbstractVector{T}, ::MOI.DualExponentialCone) where {T}
    p = projection_on_set(d, -v, MOI.ExponentialCone())
    return SVector{3,T}(v[1] + p[1], v[2] + p[2], v[3] + p[3])
end

const DEFAULT_POWER_CONE_MAX_ITERS_NEWTON = 100
const DEFAULT_POWER_CONE_MAX_ITERS_BISSECTION = 1_000
const DEFAULT_POWER_CONE_TOL_CONV = 1e-10
const DEFAULT_POWER_CONE_TOL_IN_CONE = 1e-10

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PowerCone; [max_iters_newton = 3, max_iters_bisection = 1000]) where {T}

projection of vector `v` on the power cone
i.e. `K = {(x,y,z) | x^a * y^(1-a) >= |z|, x>=0, y>=0}`.

References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Prop 2.2
"""
function projection_on_set(
    ::DefaultDistance,
    v::AbstractVector{T},
    s::MOI.PowerCone;
    max_iters_newton = DEFAULT_POWER_CONE_MAX_ITERS_NEWTON,
    max_iters_bisection = DEFAULT_POWER_CONE_MAX_ITERS_BISSECTION,
) where {T}
    _check_dimension(v, s)

    if _in_pow_cone(v, s)
        return v
    end
    if _in_pow_cone(-v, MOI.dual_set(s))
        # if in polar cone Ko = -K*
        return zeros(T, 3)
    end
    if abs(v[3]) <= DEFAULT_POWER_CONE_TOL_IN_CONE
        return [max(v[1],0), max(v[2],0), 0]
    end

    _, proj4 = _solve_system_power_cone(
        v,
        s,
        max_iters_newton = max_iters_newton,
        max_iters_bisection = max_iters_bisection,
    )
    return proj4
end

function _in_pow_cone(v::AbstractVector{T}, cone::MOI.PowerCone; tol=DEFAULT_POWER_CONE_TOL_IN_CONE) where {T}
    α = cone.exponent
    return v[1] >= 0 && v[2] >= 0 && tol + v[1]^α * v[2]^(1-α) >= abs(v[3])
end

function _in_pow_cone(v::AbstractVector{T}, cone::MOI.DualPowerCone; tol=DEFAULT_POWER_CONE_TOL_IN_CONE) where {T}
    α = cone.exponent
    return (
        v[1] >= 0 && v[2] >= 0 && tol + (v[1])^α * (v[2])^(1-α) >= α^α * (1-α)^(1-α) * abs(v[3])
    )
end

function _power_cone_system_factor(xi, αi, z, r)
    # If `xi` is negative and `r` is very close to `abs(z)`,
    # we get to a point where `√(x² + a) + x` is zero.
    # Even if the difference is small, we would like to
    # get a more accurate value than zero.
    # In order to achieve this, we use the following trick:
    # √(x² + a) + x = (x^2 + a - x^2) / √(x^2 + a) - x
    xi2 = xi^2
    a = 4*αi*r*(abs(z) - r)
    root = xi2 + a
    if xi > 0
        return xi + sqrt(root)
    else
        return a / (sqrt(root) - xi)
    end
end

function _power_cone_system(r, px, py, α)
    return 0.5 * (px^α * py^(1-α)) - r
end

function _power_cone_system(r, x, y, z, α)
    return _power_cone_system(
        r,
        _power_cone_system_factor(x, α, z, r),
        _power_cone_system_factor(y, 1 - α, z, r),
        α,
    )
end

function _solve_system_power_cone_bisection(
    v::AbstractVector{T},
    s::MOI.PowerCone;
    max_iters=DEFAULT_POWER_CONE_MAX_ITERS_BISSECTION,
    tol=DEFAULT_POWER_CONE_TOL_CONV
) where {T}
    x, y, z = v
    α = s.exponent
    # Φ is positive for r = 0 and negative for r = |z| so we can just bisect
    pos = zero(T)
    neg = abs(z)
    r = (pos + neg) / 2
    for _ in 1:max_iters
        Φ = _power_cone_system(r, x, y, z, α)
        if Φ < 0
            neg = r
        else
            pos = r
        end
        r = (pos + neg) / 2
        if abs(Φ) < tol
            break
        end
    end
    return r
end

function _solve_system_power_cone_newton(
    v::AbstractVector{T},
    s::MOI.PowerCone;
    max_iters=DEFAULT_POWER_CONE_MAX_ITERS_NEWTON,
    tol=DEFAULT_POWER_CONE_TOL_CONV
) where {T}
    x, y, z = v
    α = s.exponent
    # Solve with Newton method
    # Start Newton at |z|/2. Sol in set (0, |z|)
    dΦ_prod_dr(xi,αi,z,r,px) = 2*αi*(abs(z) - 2r) / (px - xi)
    dΦ_dr(r,phi,px,py,dpx,dpy) = (phi+r) * (α * dpx / px + (1-α) * dpy / py) - 1
    px, py = zero(T), zero(T)
    r = abs(z) * 0.46
    for _ in 1:max_iters
        if isnan(r)
            break
        end
        px = _power_cone_system_factor(x, α, z, r)
        py = _power_cone_system_factor(y, 1 - α, z, r)
        Φ = _power_cone_system(r, px, py, α)
        if abs(Φ) < tol
            break
        end
        dpx = dΦ_prod_dr(x, α, z, r, px)
        dpy = dΦ_prod_dr(y, 1 - α, z, r, py)
        dΦ = dΦ_dr(r, Φ, px, py, dpx, dpy)

        # Newton step, bounded to interval
        r = min(max(r - Φ/dΦ, 0), abs(z))
    end
    return r
end

"""
    _solve_system_power_cone(v::AbstractVector{T}, s::MOI.PowerCone) where {T}

Solves the system in [1, Proposition 2.2] to determine projection.
Returns tuple `(r, proj)`:
* `r` such that `Phi(r) = 0` and `0 < r < abs(v[3])`.
* `proj` is the projection from case 4 of the power cone

References:
[1]. [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Prop 2.2
"""
function _solve_system_power_cone(
    v::AbstractVector{T},
    s::MOI.PowerCone;
    max_iters_newton = DEFAULT_POWER_CONE_MAX_ITERS_NEWTON,
    max_iters_bisection = DEFAULT_POWER_CONE_MAX_ITERS_BISSECTION,
    tol=DEFAULT_POWER_CONE_TOL_CONV,
) where {T}
    x, y, z = v
    α = s.exponent
    # We first try to quickly get an accurate solution with Newton:
    r_newton = _solve_system_power_cone_newton(v, s; max_iters = max_iters_newton, tol)
    Φ_newton = !isnan(r_newton) ? _power_cone_system(r_newton, x, y, z, α) : NaN
    # When the optimal solution `r` is close to the boundaries `0` and `|z|`,
    # Newton has a tendency to overshoot and hit the boundary. Once the boundary
    # is hit, it will compute `NaN` so we'll need bisection to get an answer.

    Φ_bisection = Inf
    if isnan(r_newton) || abs(Φ_newton) > tol
        r_bisection = _solve_system_power_cone_bisection(v, s; max_iters = max_iters_bisection, tol)
        println(r_bisection)
        Φ_bisection = _power_cone_system(r_bisection, x, y, z, α)
        if abs(Φ_bisection) > tol
            # This happens for instance for
            # `_solve_system_power_cone([-10, 10, 1e-3], MOI.PowerCone(0.15))`
            # The value of `r` is found by the bisection to be between
            # 0.0009999999999999998
            # and
            # 0.001
            # It's not possible to do better for `Float64` but the tolerance requested
            # by the user is not satisfied so we still warn
            @warn("Error `$(abs(Φ_bisection)) > $tol` after maximum iterations hit for projection of $v onto $s")
        end
    end
    r = if isnan(r_newton) || abs(Φ_newton) > abs(Φ_bisection)
        r_bisection
    else
        r_newton
    end
    px = _power_cone_system_factor(x, α, z, r)
    py = _power_cone_system_factor(y, 1 - α, z, r)
    return r, [px / T(2), py / T(2), sign(z) * r]
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PowerCone) where {T}

projection of vector `v` on the dual power cone
i.e. `K_pow^* = {(u,v,w) | (u/a)^a * (v/(1-a))^(1-a) >= |w|, u>=0, v>=0}`.

References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Prop 2.2
"""
function projection_on_set(d::DefaultDistance, v::AbstractVector{T}, s::MOI.DualPowerCone) where {T}
    return v + projection_on_set(d, -v, MOI.PowerCone(s.exponent))
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
    return FillArrays.Eye(length(v))
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
    y = @. (-sign(v) + one(T))/2
    return LinearAlgebra.Diagonal(y)
end

"""
    projection_gradient_on_set(::NormedEpigraphDistance{p}, v::AbstractVector{T}, ::MOI.SecondOrderCone) where {T}

derivative of projection of vector `v` on second order cone i.e. K = {(t, x) ∈ R+ × Rn |  ||x|| ≤ t }

References:
* [Proximal Algorithms](https://doi.org/10.1561/2400000003), Section 6.3.2, p. 189
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
    result ./= (2 * norm_x)
    return result
end

function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.SecondOrderCone) where {T}
    return projection_gradient_on_set(NormedEpigraphDistance{2}(), v, cone)
end

function projection_gradient_on_set(d::NormedEpigraphDistance, v::AbstractVector{T}, ::MOI.RotatedSecondOrderCone) where {T}
    n = length(v)
    R = SecondOrderConeRotation()
    P = projection_gradient_on_set(d, R * v, MOI.SecondOrderCone(n))
    return R * P * R
end

function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.RotatedSecondOrderCone) where {T}
    return projection_gradient_on_set(NormedEpigraphDistance{2}(), v, cone)
end


"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, cone::MOI.PositiveSemidefiniteConeTriangle) where {T}

derivative of projection of vector `v` on positive semidefinite cone i.e. K = S^n⨥

References:
* [Proximal Algorithms](https://doi.org/10.1561/2400000003), Section 6.3.3, p. 189
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, set::MOI.PositiveSemidefiniteConeTriangle) where {T}
    n = length(v)
    X = reshape_vector(v, set)
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
        X̃ = reshape_vector(y, set)
        B = U' * X̃ * U

        for i in axes(B, 1) # do the hadamard product
            for j in axes(B, 2)
                if i <= k && j <= k
                    @inbounds B[i, j] = 0
                elseif i > k && j <= k
                    λpi = max(λ[i], zero(Tp))
                    λmj = -min(λ[j], zero(Tp))
                    @inbounds B[i, j] *= λpi / (λmj + λpi)
                elseif i <= k && j > k
                    λmi = -min(λ[i], zero(Tp))
                    λpj = max(λ[j], zero(Tp))
                    @inbounds B[i, j] *= λpj / (λmi + λpj)
                end
            end
        end
        @inbounds D[idx, :] = vectorize(LinearAlgebra.Symmetric(U * B * U'))
        # reset eigenvector
        @inbounds y[idx] = 0
    end
    return D
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, set::MOI.Scaled) where {T}

derivative of projection of vector `v` on the scaled version of `set.set`.
"""
function projection_gradient_on_set(d::DefaultDistance, v::AbstractVector{T}, set::MOI.Scaled) where {T}
    scale = MOI.Utilities.SetDotScalingVector{T}(set.set)
    D = LinearAlgebra.Diagonal(scale)
    # ∂(D⋅p(D⁻¹⋅v))/∂v = ∂(D⋅p(D⁻¹⋅v))/∂(D⁻¹⋅v) ⋅ ∂(D⁻¹⋅v)/∂v
    #                 = D ⋅ ∂(p(D⁻¹⋅v))/∂(D⁻¹⋅v) ⋅ D⁻¹
    return D * projection_gradient_on_set(d, D \ v, set.set) / D
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.ExponentialCone) where {T}

derivative of projection of vector `v` on closure of the exponential cone,
i.e. `cl(Kexp) = {(x,y,z) | y e^(x/y) <= z, y>0 } U {(x,y,z)| x <= 0, y = 0, z >= 0}`.

References:
* [Solution Refinement at Regular Points of Conic Problems](https://stanford.edu/~boyd/papers/cone_prog_refine.html)
by Enzo Busseti, Walaa M. Moursi, and Stephen Boyd
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.ExponentialCone) where {T}
    _check_dimension(v, s)

    if _in_exp_cone(v; dual=false)
        return SMatrix{3,3,T}(I)
    end
    if _in_exp_cone(-v; dual=true)
        # if in polar cone Ko = -K*
        return zeros(SMatrix{3,3,T})
    end
    if v[1] <= 0 && v[2] <= 0
        return @SMatrix(T[
            1 0 0
            0 0 0
            0 0 (v[3] >= 0)
        ])
    end

    # in case of Friberg heuristic solutions
    ret, mat_h = _exp_cone_heuristic_projection_grad(v)
    if ret
        return mat_h
    end

    z1, z2, z3 = _exp_cone_proj_case_4(v)
    nu = z3 - v[3]
    rs = z1/z2
    exp_rs = exp(rs)

    mat = inv(@SMatrix([
        1+nu*exp_rs/z2     -nu*exp_rs*rs/z2       0     exp_rs;
        -nu*exp_rs*rs/z2   1+nu*exp_rs*rs^2/z2    0     (1-rs)*exp_rs;
        0                  0                      1     -1
        exp_rs             (1-rs)*exp_rs          -1    0
    ]))
    return SMatrix{3,3}(@view(mat[1:3,1:3]))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.DualExponentialCone) where {T}

derivative of projection of vector `v` on the dual exponential cone,
i.e. `Kexp^* = {(u,v,w) | u < 0, -u*exp(v/u) <= exp(1)*w } U {(u,v,w)| u == 0, v >= 0, w >= 0}`.

References:
* [Solution Refinement at Regular Points of Conic Problems]
(https://stanford.edu/~boyd/papers/cone_prog_refine.html)
by Enzo Busseti, Walaa M. Moursi, and Stephen Boyd
"""
function projection_gradient_on_set(d::DefaultDistance, v::AbstractVector{T}, ::MOI.DualExponentialCone) where {T}
    # from Moreau decomposition: x = P_K(x) + P_-K*(x)
    return I - projection_gradient_on_set(d, -v, MOI.ExponentialCone())
end

"""
    projection_gradient_on_set(d::DefaultDistance, v::AbstractVector{T}, ::MOI.PowerCone) where {T}

derivative of projection of vector `v` on the power cone
i.e. `K = {(x,y,z) | x^a * y^(1-a) >= |z|, x>=0, y>=0}`.

References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Theorem 3.1
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.PowerCone) where {T}
    _check_dimension(v, s)

    if _in_pow_cone(v, s)
        return Matrix{T}(I, 3, 3)
    end
    if _in_pow_cone(-v, MOI.dual_set(s))
        # if in polar cone Ko = -K*
        return zeros(T, 3, 3)
    end
    if abs(v[3]) <= DEFAULT_POWER_CONE_TOL_IN_CONE
        return _pow_cone_∇proj_case_3(v, s)
    end

    x, y, z = v
    α = s.exponent
    # # This handles the case where v[3]/norm(v) is small.
    #   Phi(eps()) ≈ -r
    #   Phi_prod(x, α, z, eps()) ≈ x + |x| = 2*max(x, 0)
    #   => this gets to case 3 in the limit
    # if (x < 0 || y < 0) && abs(z) <= 1e-2*norm(v)*(0.5 - abs(0.5 - α))
    #     r = eps()
    # else
    #     r, _ = _solve_system_power_cone(v, s)
    # end

    r, _ = _solve_system_power_cone(v, s)
    za = abs(z)
    gx = sqrt(x^2 + 4*α*r*(za - r))
    gy = sqrt(y^2 + 4*(1-α)*r*(za - r))
    fx = 0.5*(x + gx)
    fy = 0.5*(y + gy)

    β = 1-α
    K = -(α*x/gx + β*y/gy)
    L = 2*(za - r) / (za + (za - 2r) * -K)
    J_ii(w, γ, g) = 0.5 + w/2g + γ^2*(za - 2r)*r*L/g^2
    J_ij = α*β*(za - 2r)*r*L/(gx*gy)
    J = [
        J_ii(x, α, gx)      J_ij                sign(z)*α*r*L/gx;
        J_ij                J_ii(y, β, gy)      sign(z)*β*r*L/gy;
        sign(z)*α*r*L/gx    sign(z)*β*r*L/gy    r/za*(1+K*L)
    ]
    return J
end

"""
References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Theorem 3.1
eq (11)
"""
function _pow_cone_∇proj_case_3(v::AbstractVector{T}, s::MOI.PowerCone) where {T}
    x = [v[1]; v[2]]
    αs = [s.exponent; 1-s.exponent]

    if sum(αs[x .> 0]) > sum(αs[x .< 0])
        d = 1
    elseif sum(αs[x .> 0]) < sum(αs[x .< 0])
        d = 0
    else
        num = reduce(*, (-x[x .< 0]).^αs[x .< 0])
        denom = reduce(*, x[x .> 0].^αs[x .> 0]) * reduce(*, αs[x .< 0].^αs[x .< 0])
        d = 1/((num/denom)^2 + 1)
    end
    return LinearAlgebra.diagm(0 => T[v[1] > 0, v[2] > 0, d])
end

"""
    projection_gradient_on_set(d::DefaultDistance, v::AbstractVector{T}, ::MOI.DualPowerCone) where {T}

derivative of projection of vector `v` on the dual power cone
i.e. `K_pow^* = {(u,v,w) | (u/a)^a * (v/(1-a))^(1-a) >= |w|, u>=0, v>=0}`.

References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Theorem 3.1
"""
function projection_gradient_on_set(d::DefaultDistance, v::AbstractVector{T}, s::MOI.DualPowerCone) where {T}
    # from Moreau decomposition: x = P_K(x) + P_-K*(x)
    return I - projection_gradient_on_set(d, -v, MOI.dual_set(s))
end

"""
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::AbstractVector{<:MOI.AbstractSet})

Derivative of the projection of vector `v` on product of `sets`
projection_gradient_on_set[i,j] = ∂projection_on_set[i] / ∂v[j] where `projection_on_set` denotes projection of `v` on `cone`

Find expression of projections on cones and their derivatives here: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::AbstractVector{<:MOI.AbstractSet}) where {T}
    length(v) == length(sets) || throw(DimensionMismatch("Mismatch between value and set"))
    return BlockDiagonal([projection_gradient_on_set(DefaultDistance(), v[i], sets[i]) for i in eachindex(sets)])
end

"""
    projection_on_set(::DefaultDistance, V::AbstractVector{T}, s::NormBallNuclear{R}) where {T, R}

projection of matrix `V` onto nuclear norm ball
"""
function projection_on_set(d::DefaultDistance, V::AbstractMatrix{T}, s::NormNuclearBall{R}) where {T, R}
    U, sing_val, Vt = LinearAlgebra.svd(V)
    if (sum(sing_val) <= s.radius)
        return V
    end
    sing_val_proj = projection_on_set(d, sing_val, ProbabilitySimplex(length(sing_val), s.radius))
    return U * Diagonal(sing_val_proj) * Vt'
end

# initial implementation in FrankWolfe.jl
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::ProbabilitySimplex{R}) where {T, R}
    _check_dimension(v, s)
    # TODO: allocating a ton, should implement the recent non-sorting alg
    n = length(v)
    if sum(v) ≈ s.radius && all(>=(0), v)
        return v
    end
    rev = v .- maximum(v)
    u = sort(rev, rev=true)
    cssv = cumsum(u)
    rho = sum(eachindex(u)) do idx
        u[idx] * idx > (cssv[idx] - s.radius)
    end - 1
    theta = (cssv[rho+1] - s.radius) / (rho + 1)
    w = clamp.(rev .- theta, 0.0, Inf)
    return w
end

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::StandardSimplex{R}) where {T, R}
    _check_dimension(v, s)
    n = length(v)
    if sum(v) ≤ s.radius && all(>=(0), v)
        return v
    end
    x = copy(v)
    sum_pos = zero(T)
    for idx in eachindex(x)
        if x[idx] < 0
            x[idx] = 0
        else
            sum_pos += x[idx]
        end
    end
    # at least one positive element
    if sum_pos > 0
        @. x = x / sum_pos * s.radius
    end
    return x
end

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormInfinityBall{R}) where {T, R}
    if norm(v, Inf) <= s.radius
        return v
    end
    return clamp.(v, -s.radius, s.radius)
end

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormTwoBall{R}) where {T, R}
    nv = norm(v)
    if nv <= s.radius
        return v
    end
    return v .* s.radius ./ nv
end

# inspired by https://github.com/MPF-Optimization-Laboratory/ProjSplx.jl
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormOneBall{R}) where {T, R}
    TP = promote_type(T, R)
    n = length(v)
    τ = s.radius
    if norm(v, 1) ≤ τ
        return v
    end
    u = TP.(abs.(v))
    # simplex projection
    bget = false
    s_indices = sortperm(u, rev=true)
    tsum = zero(T)

    @inbounds for i in 1:n-1
        tsum += u[s_indices[i]]
        tmax = (tsum - τ) / i
        if tmax ≥ u[s_indices[i+1]]
            bget = true
            break
        end
    end
    if !bget
        tmax = (tsum + u[s_indices[n]] - τ) / n
    end

    @inbounds for i in 1:n
        u[i] = max(u[i] - tmax, 0)
        u[i] *= sign(v[i])
    end
    return u
end
