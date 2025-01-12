using Test
import MathOptInterface as MOI

using MathOptSetDistances
const MOD = MathOptSetDistances

using FiniteDifferences
using LinearAlgebra

const bfdm = FiniteDifferences.backward_fdm(5, 1)
const ffdm = FiniteDifferences.forward_fdm(5, 1)
const cfdm = FiniteDifferences.central_fdm(5,1)

import ChainRulesCore as CRC
import FillArrays

# type piracy because of https://github.com/JuliaDiff/FiniteDifferences.jl/issues/177
function FiniteDifferences.to_vec(x::FillArrays.Zeros)
    v = collect(x)
    return v, _ -> error("can't create `Zeros` from a vector")
end

"""
A multivariate Gaussian generator without points too close to 0
"""
function safe_randn(n)
    v = 2 * randn(n)
    for i in eachindex(v)
        while v[i] ≈ 0
            v[i] = 2 * randn()
        end
    end
    return v
end


"""
A helper function for better errors on projection tests
"""
function _test_projection(v, set, dΠ, grad_fdm1, grad_fdm2, tol)
    good1 = ≈(dΠ, grad_fdm1, atol=tol)
    good2 = ≈(dΠ, grad_fdm2, atol=tol)
    if good1 || good2
        return true
    end
    error(
        """
        input:
        v   = $v
        set = $set
        tol = $tol
        projections:
        dΠ        = $dΠ
        grad_fdm1 = $grad_fdm1
        grad_fdm2 = $grad_fdm2
        """
    )
end

@testset "Test gradients with finite differences" begin
    Ntrials = 10
    @testset "Dimension $n" for n in (1, 3, 10)
        vector_sets = (
            MOI.Zeros(n),
            MOI.Reals(n),
            MOI.Nonnegatives(n),
            MOI.Nonpositives(n),
        )
        vs = [safe_randn(n) for _ in 1:Ntrials]
        for _ in 1:Ntrials
            push!(vs, rand(-5:2:5, n))
        end
        @testset "Vector set $s" for s in vector_sets
            @testset "Vector $v" for v in vs
                dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                grad_fdm1 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                grad_fdm2 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                # finite diff can mess up for low v
                @test all(eachindex(v)) do idx
                    dΠ[idx,idx] ≈ grad_fdm1[idx,idx] || dΠ[idx,idx] ≈ grad_fdm2[idx,idx]
                end
            end
        end
        @testset "PSD cone" begin
            s = MOI.PositiveSemidefiniteConeTriangle(n)
            scale = MOI.Utilities.SetDotScalingVector{Float64}(s)
            D = LinearAlgebra.Diagonal(scale)
            for _ in 1:Ntrials
                L = 3 * tril(rand(n, n))
                M = L * L'
                v = MOD.vectorize(LinearAlgebra.Symmetric(M))
                @testset "Positive definite" begin
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test dΠ ≈ I
                end
                @testset "Scaled positive definite" begin
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), D * v, MOI.Scaled(s))
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), D * v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), D * v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test dΠ ≈ I
                end
                @testset "Negative definite" begin
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), -v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), -v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), -v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    if !isapprox(det(M), 0, atol=10e-6)
                        @test all(dΠ .≈ 0)
                    end
                end
                @testset "Scaled negative definite" begin
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), -D * v, MOI.Scaled(s))
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), -D * v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), -D * v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    if !isapprox(det(M), 0, atol=10e-6)
                        @test all(dΠ .≈ 0)
                    end
                end
            end
        end
        @testset "SOC" begin
            s = MOI.SecondOrderCone(n+1)
            for _ in 1:Ntrials
                x = safe_randn(n)
                @testset "SOC interior and negative bound" begin
                    t = LinearAlgebra.norm2(x) + 2 * rand()
                    v = vcat(t, x)
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test dΠ ≈ grad_fdm1 || dΠ ≈ grad_fdm2
                    v = vcat(-t, x)
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test dΠ ≈ grad_fdm1 || dΠ ≈ grad_fdm2
                end
                @testset "Out of cone point" begin
                    for tscale in (0.1, 0.5, 0.9)
                        t = tscale * LinearAlgebra.norm2(x)
                        v = vcat(t, x)
                        dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                        grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                        grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                        @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                        @test dΠ ≈ grad_fdm1 || dΠ ≈ grad_fdm2
                        t = tscale
                        xr = x / norm(x)
                        v = vcat(t, xr)
                        dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                        # theoretical expression for unit vector
                        @test 2dΠ ≈ [
                            1 xr'
                            xr ((t + 1) * I - t * xr * xr')
                        ]
                    end
                end
            end
        end
        @testset "RSOC" begin
            s = MOI.RotatedSecondOrderCone(n+2)
            for _ in 1:Ntrials
                x = safe_randn(n)
                @testset "SOC interior and negative bound" begin
                    t = LinearAlgebra.norm2(x) + 2 * rand()
                    u = 1/2
                    for (t, u) in [(t^2, 1/2), (1/2, t^2), (t, t/2), (t/2, t)]
                        for u in [u, -u]
                            for v in [t, -t]
                                v = vcat(t, u, x)
                                dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                                grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                                grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                                @test dΠ ≈ grad_fdm1 || dΠ ≈ grad_fdm2
                            end
                        end
                    end
                end
                @testset "Out of cone point" begin
                    for tscale in (0.1, 0.5, 0.9)
                        t = tscale * LinearAlgebra.norm2(x)
                        for (t, u) in [(t^2, 1/2), (1/2, t^2), (t, t/2), (t/2, t)]
                            for u in [u, -u]
                                for v in [t, -t]
                                    v = vcat(t, x)
                                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                                    @test dΠ ≈ grad_fdm1 || dΠ ≈ grad_fdm2
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    @testset "Indefinite matrix" begin
        s = MOI.PositiveSemidefiniteConeTriangle(2)
        scale = MOI.Utilities.SetDotScalingVector{Float64}(s)
        D = LinearAlgebra.Diagonal(scale)
        Q = [
            1 0
            0 -1
        ]
        Qi = Q'
        B = [
            0 1/2
            1/2 1
        ]
        for _ in 1:5
            # scale factor
            f = 20 * rand() + 5
            A = [
                -f 0
                0 f
            ]
            Λ = Diagonal([-f, f])
            Λp = Diagonal([0, f])
            @test A ≈ Q * Λ * Qi
            v = MOD.vectorize(LinearAlgebra.Symmetric(A))
            Πv = MOD.projection_on_set(MOD.DefaultDistance(), v, s)
            Π = MOD.reshape_vector(Πv, s)
            @test Π ≈ Q * Λp * Qi
            scaled_Πv = MOD.projection_on_set(MOD.DefaultDistance(), D * v, MOI.Scaled(s))
            scaled_Π = MOD.reshape_vector(D \ scaled_Πv, s)
            @test scaled_Π ≈ Q * Λp * Qi
            DΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
            scaled_DΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), D * v, MOI.Scaled(s))
            # directional derivative
            for _ in 1:Ntrials
                Xd = randn(2,2)
                xd = MOD.vectorize(LinearAlgebra.Symmetric(Xd))
                QBX = Q * (B .* (Q' * Xd * Q)) * Q'
                @test DΠ * xd ≈ MOD.vectorize(LinearAlgebra.Symmetric(QBX))
                @test scaled_DΠ * D * xd ≈ D * MOD.vectorize(LinearAlgebra.Symmetric(QBX))
            end
            grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
            grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
            @test grad_fdm1 ≈ DΠ || grad_fdm2 ≈ DΠ
            scaled_grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), D * v)[1]'
            scaled_grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, MOI.Scaled(s)), D * v)[1]'
            @test scaled_grad_fdm1 ≈ scaled_DΠ || scaled_grad_fdm2 ≈ scaled_DΠ
        end
    end
    @testset "Scalar $ST" for ST in (MOI.LessThan, MOI.GreaterThan, MOI.EqualTo)
        s = ST(10 * randn())
        for v in 1:Ntrials
            v = randn()
            while v ≈ MOI.constant(s)
                v += 2 * randn()
            end
            dfor = ffdm(x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)
            dback = bfdm(x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)
            dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
            @test ≈(dfor, dΠ, atol=1e-5) || ≈(dback, dΠ, atol=1e-5)
        end
    end

    @testset "Exp Cone" begin
        function det_case_exp_cone(v; dual=false)
            v = dual ? -v : v
            if MOD._in_exp_cone(v; dual=false)
                return 1
            elseif MOD._in_exp_cone(-v; dual=true)
                return 2
            elseif v[1] <= 0 && v[2] <= 0 #TODO: threshold here??
                return 3
            else
                return 4
            end
        end

        rng = Random.MersenneTwister(0)
        # Random.seed!(0)
        s = MOI.ExponentialCone()
        sd = MOI.DualExponentialCone()
        case_p = zeros(4)
        case_d = zeros(4)
        # Adjust tolerance down because a 1-2 errors when projection ends up
        # very close to the z axis
        # For intuition, see Fig 5.1 https://docs.mosek.com/modeling-cookbook/expo.html
        #   Note that their order is reversed: (x, y, z) = (x3, x2, x1) [theirs]
        tol = 1e-4
        for ii in 1:100
            # v = 5*randn(3)
            v = 5*randn(rng, 3)
            @testset "Primal Cone" begin
                case_p[det_case_exp_cone(v; dual=false)] += 1
                dΠ = @inferred MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                @test _test_projection(v, s, dΠ, grad_fdm1, grad_fdm2, tol)
            end

            @testset "Dual Cone" begin
                case_d[det_case_exp_cone(v; dual=true)] += 1
                dΠ = @inferred MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, sd)
                grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                @test _test_projection(v, sd, dΠ, grad_fdm1, grad_fdm2, tol)
            end
        end
        list_of_points = [
            [0.04, -3, 11],  # https://github.com/matbesancon/MathOptSetDistances.jl/issues/63
            -[0.04, -3, 11],
            [0.01, -1, 1], # Friberg heuristic case 1
            -[0.01, -1, 1],
            [0, 1, 10], # Friberg heuristic case 2
            -[0, 1, 10],
            [0, 1, 0.5], # Friberg heuristic case 3
            -[0, 1, 0.5],
        ]
        for v in list_of_points
            @testset "Primal Cone" begin
                case_p[det_case_exp_cone(v; dual=false)] += 1
                dΠ = @inferred MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                @test _test_projection(v, s, dΠ, grad_fdm1, grad_fdm2, tol)
            end

            @testset "Dual Cone" begin
                case_d[det_case_exp_cone(v; dual=true)] += 1
                dΠ = @inferred MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, sd)
                grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                @test _test_projection(v, sd, dΠ, grad_fdm1, grad_fdm2, tol)
            end
        end
        @test all(case_p .> 0) && all(case_d .> 0)
    end

    @testset "Power Cone" begin
        function det_case_pow_cone(x, α; dual=false)
            v = dual ? -x : x
            s = MOI.PowerCone(α)
            if MOD._in_pow_cone(v, s)
                return 1
            elseif MOD._in_pow_cone(-v, MOI.dual_set(s))
                return 2
            elseif abs(v[3]) <= 1e-8
                return 3
            else
                return 4
            end
        end


        case_p = zeros(4)
        case_d = zeros(4)

        rng = Random.MersenneTwister(0)
        # review fails on power cone gradient
        Random.seed!(0)
        tol = 1e-4
        for ii in 1:100
            v = 5*randn(3)
            # v = 5*randn(rng, 3)
            for α in [0.5; rand(0.05:0.05:0.95)]
            # for α in [0.5; rand(rng, 0.05:0.05:0.95)]
                if ii % 10 == 1
                    v[3] = 0.0
                end
                s = MOI.PowerCone(α)
                sd = MOI.dual_set(s)
                @testset "Primal Cone" begin
                    case = det_case_pow_cone(v, α; dual=false)
                    case_p[case] += 1
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm3 = FiniteDifferences.jacobian(cfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    ##
                    # References:
                    # * [Differential properties of Euclidean projection onto power cone]
                    # (https://link.springer.com/article/10.1007/s00186-015-0514-0), Theorem 3.1
                    # eq (11)
                    x = [v[1]; v[2]]
                    αs = [s.exponent; 1-s.exponent]
                    d = if sum(αs[x .> 0]) > sum(αs[x .< 0])
                        1
                    elseif sum(αs[x .> 0]) < sum(αs[x .< 0])
                        0
                    else
                        NaN
                    end
                    if v[3] == 0 && !isnan(d)
                        grad_fdm1[end] = d
                        grad_fdm2[end] = d
                        grad_fdm3[end] = d
                    end
                    ##
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test ≈(dΠ, grad_fdm1,atol=tol) || ≈(dΠ, grad_fdm2, atol=tol) || ≈(dΠ, grad_fdm3, atol=tol)
                    if !(≈(dΠ, grad_fdm1,atol=tol) || ≈(dΠ, grad_fdm2, atol=tol) || ≈(dΠ, grad_fdm3, atol=tol))
                        @show MOD._pow_cone_∇proj_case_3(v, s)
                        error("α=$α\nv=$v\ndΠ = $dΠ\ncase=$case\nFD1=$grad_fdm1\nFD2=$grad_fdm2\nFD3=$grad_fdm3")
                    end
                end

                @testset "Dual Cone" begin
                    case = det_case_pow_cone(v, α; dual=true)
                    case_d[case] += 1
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, sd)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                    grad_fdm3 = FiniteDifferences.jacobian(cfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, sd), v)[1]'
                    ##
                    # References:
                    # * [Differential properties of Euclidean projection onto power cone]
                    # (https://link.springer.com/article/10.1007/s00186-015-0514-0), Theorem 3.1
                    # eq (11)
                    x = [v[1]; v[2]]
                    αs = [s.exponent; 1-s.exponent]
                    d = if sum(αs[x .> 0]) > sum(αs[x .< 0])
                        1
                    elseif sum(αs[x .> 0]) < sum(αs[x .< 0])
                        0
                    else
                        NaN
                    end
                    if v[3] == 0 && !isnan(d)
                        grad_fdm1[end] = d
                        grad_fdm2[end] = d
                        grad_fdm3[end] = d
                    end
                    ##
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    @test ≈(dΠ, grad_fdm1,atol=tol) || ≈(dΠ, grad_fdm2, atol=tol) || ≈(dΠ, grad_fdm3, atol=tol)
                    if !(≈(dΠ, grad_fdm1,atol=tol) || ≈(dΠ, grad_fdm2, atol=tol) || ≈(dΠ, grad_fdm3, atol=tol))
                        error("v=$v\ndΠ = $dΠ\ncase=$case\nFD1=$grad_fdm1\nFD2=$grad_fdm2\nFD3=$grad_fdm3")
                    end
                end
            end
        end
        @test all(case_p .> 0) && all(case_d .> 0)
    end
end
