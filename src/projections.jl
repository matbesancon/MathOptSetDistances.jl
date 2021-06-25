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
    return X[LinearAlgebra.triu(trues(size(X)))]
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
(https://docs.mosek.com/whitepapers/expcone-proj.pdf) by Henrik Friberg
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

function _exp_cone_proj_case_4(v::AbstractVector{T}; tol=1e-8) where {T}
    # Try Heuristic solutions [Friberg 2021, Lemma 5.1]
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
i.e. `Kexp^* = {(u,v,w) | u < 0, -u*exp(v/u) <= ew } U {(u,v,w)| u == 0, v >= 0, w >= 0}`.

References:
* [Solution Refinement at Regular Points of Conic Problems]
(https://stanford.edu/~boyd/papers/cone_prog_refine.html)
by Enzo Busseti, Walaa M. Moursi, and Stephen Boyd
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.DualExponentialCone) where {T}
    # from Moreau decomposition: x = P_K(x) + P_-K*(x)
    return I - projection_gradient_on_set(DefaultDistance(), -v, MOI.ExponentialCone())
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
    projection_on_set(::DefaultDistance, V::AbstractVector{T}, s::NormBallNuclear{T}) where {T}

projection of matrix `V` onto nuclear norm ball
"""
function projection_on_set(d::DefaultDistance, V::AbstractMatrix{T}, s::NormNuclearBall{T}) where {T}
    U, sing_val, Vt = LinearAlgebra.svd(V)
    if (sum(sing_val) <= s.radius)
        return V
    end
    sing_val_proj = projection_on_set(d, sing_val, ProbabilitySimplex(length(sing_val), s.radius))
    return U * Diagonal(sing_val_proj) * Vt'
end

# initial implementation in FrankWolfe.jl
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::ProbabilitySimplex{T}) where {T}
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

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::StandardSimplex{T}) where {T}
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

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormInfinityBall{T}) where {T}
    if norm(v, Inf) <= s.radius
        return v
    end
    return clamp.(v, -s.radius, s.radius)
end

function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormTwoBall{T}) where {T}
    nv = norm(v)
    if nv <= s.radius
        return v
    end
    return v .* s.radius ./ nv
end

# inspired by https://github.com/MPF-Optimization-Laboratory/ProjSplx.jl
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::NormOneBall{T}) where {T}
    n = length(v)
    if norm(v, 1) ≤ τ
        return v
    end
    u = abs.(v)
    # simplex projection
    bget = false
    s_indices = sortperm(u, rev=true)
    tsum = zero(τ)

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
