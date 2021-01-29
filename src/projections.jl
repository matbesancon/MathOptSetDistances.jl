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
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.ExponentialCone) where {T}
    _check_dimension(v, s)

    if _in_exp_cone(v; dual=false)
        return v
    end
    if _in_exp_cone(-v; dual=true)
        # if in polar cone Ko = -K*
        return zeros(T, 3)
    end
    if v[1] <= 0 && v[2] <= 0
        return [v[1]; 0.0; max(v[3],0)]
    end

    return _exp_cone_proj_case_4(v)
end

function _in_exp_cone(v::AbstractVector{T}; dual=false, tol=1e-8) where {T}
    if dual
        return (
            (isapprox(v[1], 0, atol=tol) && v[2] >= 0 && v[3] >= 0) ||
            (v[1] < 0 && v[1]*exp(v[2]/v[1]) + ℯ*v[3] >= tol)
        )
    else
        return (
            (v[1] <= 0 && isapprox(v[2], 0, atol=tol) && v[3] >= 0) ||
            (v[2] > 0 && v[2] * exp(v[1] / v[2]) - v[3] <= tol)
        )
    end
end

function _exp_cone_proj_case_4(v::AbstractVector{T}) where {T}
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

    x, code = _bisection(h, lb, ub)
    if code > 0
        error("Failured to solve root finding problem in exp cone projection")
    end

    return ((x - 1)*r + s)/(x^2 - x + 1) * [x; 1.0; exp(x)]
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
    return v + projection_on_set(d, -v, MOI.ExponentialCone())
end

"""
    projection_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.PowerCone) where {T}

projection of vector `v` on the power cone
i.e. `K = {(x,y,z) | x^a * y^(1-a) >= |z|, x>=0, y>=0}`.

References:
* [Differential properties of Euclidean projection onto power cone]
(https://link.springer.com/article/10.1007/s00186-015-0514-0), Prop 2.2
"""
function projection_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.PowerCone) where {T}
    _check_dimension(v, s)

    if _in_pow_cone(v, s)
        return v
    end
    if _in_pow_cone(-v, MOI.dual_set(s))
        # if in polar cone Ko = -K*
        return zeros(T, 3)
    end
    if abs(v[3]) <= 1e-10
        return [max(v[1],0); max(v[2],0); 0.0]
    end

    return _solve_system_pow_cone(v, s; arg=false)
end

function _in_pow_cone(v::AbstractVector{T}, cone::MOI.PowerCone; tol=1e-10) where {T}
    α = cone.exponent
    return v[1] >= 0 && v[2] >= 0 && tol + v[1]^α * v[2]^(1-α) >= abs(v[3])
end

function _in_pow_cone(v::AbstractVector{T}, cone::MOI.DualPowerCone; tol=1e-10) where {T}
    α = cone.exponent
    return (
        v[1] >= 0 && v[2] >=0 && tol + (v[1])^α * (v[2])^(1-α) >= α^α * (1-α)^(1-α) * abs(v[3])
    )
end

function _solve_system_pow_cone(v::AbstractVector{T}, s::MOI.PowerCone; arg=true) where {T}
    x, y, z = v
    α = s.exponent
    Phi_prod(xi,αi,z,r) = (xi + sqrt(xi^2 + 4*αi*r*(abs(z) - r)))
    Phi(r) = 0.5*(Phi_prod(x,α,z,r)^α * Phi_prod(y,1-α,z,r)^(1-α)) - r

    # Search for initial upper an lower bounds. Sol in set (0, |z|)
    lb, ub = eps(), abs(z) - eps()
    ii = 1
    while sign(Phi(lb)) == sign(Phi(ub))
        lb += 10^ii * 1e-15
        ub -= 10^ii * 1e-15
        ub < lb && error("Bound interval error. v = $v, α = $α")
        ii+=1
    end
    r, code = _bisection(Phi, lb, ub)
    if code > 0
        error("Failured to solve root finding problem in power cone projection")
    end
    return arg ? r : [0.5*Phi_prod(x,α,z,r); 0.5*Phi_prod(y,1-α,z,r); sign(z)*r]
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
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, ::MOI.ExponentialCone) where {T}

derivative of projection of vector `v` on closure of the exponential cone,
i.e. `cl(Kexp) = {(x,y,z) | y e^(x/y) <= z, y>0 } U {(x,y,z)| x <= 0, y = 0, z >= 0}`.

References:
* [Solution Refinement at Regular Points of Conic Problems](https://stanford.edu/~boyd/papers/cone_prog_refine.html)
by Enzo Busseti, Walaa M. Moursi, and Stephen Boyd
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, s::MOI.ExponentialCone) where {T}
    _check_dimension(v, s)
    Ip(z) = z >= 0 ? 1 : 0

    if _in_exp_cone(v; dual=false)
        return Matrix{T}(I, 3, 3)
    end
    if _in_exp_cone(-v; dual=true)
        # if in polar cone Ko = -K*
        return zeros(T, 3, 3)
    end
    if v[1] <= 0 && v[2] <= 0
        return LinearAlgebra.diagm(0 => T[1, Ip(v[2]), Ip(v[3])])
    end

    z1, z2, z3 = _exp_cone_proj_case_4(v)
    nu = z3 - v[3]
    rs = z1/z2
    exp_rs = exp(rs)

    mat = inv([
        1+nu*exp_rs/z2     -nu*exp_rs*rs/z2       0     exp_rs;
        -nu*exp_rs*rs/z2   1+nu*exp_rs*rs^2/z2    0     (1-rs)*exp_rs;
        0                  0                      1     -1
        exp_rs             (1-rs)*exp_rs          -1    0
    ])
    return mat[1:3,1:3]
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
    if abs(v[3]) <= 1e-8
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
    #     r = _solve_system_pow_cone(v, s; arg=true)
    # end

    r = _solve_system_pow_cone(v, s; arg=true)
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
    projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet})

Derivative of the projection of vector `v` on product of `sets`
projection_gradient_on_set[i,j] = ∂projection_on_set[i] / ∂v[j] where `projection_on_set` denotes projection of `v` on `cone`

Find expression of projections on cones and their derivatives here: https://stanford.edu/~boyd/papers/pdf/cone_prog_refine.pdf
"""
function projection_gradient_on_set(::DefaultDistance, v::AbstractVector{T}, sets::Array{<:MOI.AbstractSet}) where {T}
    length(v) == length(sets) || throw(DimensionMismatch("Mismatch between value and set"))
    return BlockDiagonal([projection_gradient_on_set(DefaultDistance(), v[i], sets[i]) for i in eachindex(sets)])
end
