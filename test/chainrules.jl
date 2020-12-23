using ChainRulesCore
const CRC = ChainRulesCore
using ChainRulesTestUtils

import MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
using FiniteDifferences
using Test
using Random

# avoid random finite diff fails because of rounding
Random.seed!(42)

"""
Used to test all distances where the set does not take differentiable parameters.
"""
function test_rrule_analytical(x, s; distance = MOD.DefaultDistance(), ntrials = 10, rng=Random.GLOBAL_RNG, atol=1e-4,rtol=1e-4, test_fdiff=true)
    y = MOD.projection_on_set(distance, x, s)
    dΠ = MOD.projection_gradient_on_set(distance, x, s)
    (yprimal, pullback) = CRC.rrule(MOD.projection_on_set, distance, x, s)
    @test yprimal ≈ y
    for _ in 1:ntrials
        xb = randn(rng, length(x))
        yb = randn(rng, length(x))
        (_, _, Δx, _) = pullback(yb)
        @test Δx ≈ dΠ' * yb
        if test_fdiff
            ChainRulesTestUtils.rrule_test(
                MOD.projection_on_set,
                yb,
                (distance, nothing),
                (x, xb),
                (s, nothing),
                atol=atol,rtol=rtol
            )
        end
    end
end

@testset "rrules multivariate" begin
    for n in (1, 2, 10)
        x = randn(n)
        s = MOI.Reals(n)
        test_rrule_analytical(x, s, atol=1e-5, rtol=1e-5)
        s = MOI.Zeros(n)
        test_rrule_analytical(x, s, atol=1e-5, rtol=1e-5, test_fdiff=false)
        # requires FillArrays.Zero handling
        # still broken?
        @test_broken ChainRulesTestUtils.rrule_test(
            MOD.projection_on_set,
            yb,
            (MOD.DefaultDistance(), nothing),
            (x, xb),
            (s, nothing),
        )
        @testset "Orthant $s" for s in (MOI.Nonpositives(n), MOI.Nonnegatives(n))
            test_rrule_analytical(x, s)
        end
        @testset "SOC $n" begin
            s = MOI.SecondOrderCone(n+1)
            for _ in 1:10
                x = 10 * randn(n+1)
                if norm(x[2:end]) ≈ x[1]
                    if rand() > 0.8
                        x[1] *= 2
                    else
                        x[1] /= 2
                    end
                end
                test_rrule_analytical(x, s)
            end
        end
    end
end

@testset "rrules univariate" begin
    # note: we do not use the generic test function since sets have derivatives
    @testset "Scalar $ST" for ST in (MOI.EqualTo, MOI.LessThan, MOI.GreaterThan)
        for _ in 1:10
            s = ST(10 * randn())
            x = 10 * randn()
            # avoid non-differentiable points
            if isapprox(x, MOI.constant(s), atol=1e-5)
                δ = 2rand()
                if rand(Bool)
                    x += δ
                else
                    x -= δ
                end
            end
            dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), x, s)
            y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
            (yprimal, pullback) = CRC.rrule(MOD.projection_on_set, MOD.DefaultDistance(), x, s)
            @test yprimal ≈ y
            for _ in 1:5
                xb = ChainRulesTestUtils.rand_tangent(x)
                yb = ChainRulesTestUtils.rand_tangent(y)
                sb = ChainRulesTestUtils.rand_tangent(s)
                ChainRulesTestUtils.rrule_test(
                    MOD.projection_on_set,
                    yb,
                    (MOD.DefaultDistance(), nothing),
                    (x, xb),
                    (s, sb),
                    atol=1e-4,
                )
                (_, _, Δx, _) = pullback(yb)
                @test Δx ≈ dΠ' * yb
            end
        end
    end
end

@testset "frule" begin
    d = MOD.DefaultDistance()
    for n in (1, 2, 10)
        s = MOI.PositiveSemidefiniteConeTriangle(n)
        @testset "$s" begin
            for _ in 1:5
                L = 3 * tril(rand(n, n))
                M = L * L'
                v0 = MOD.vec_symm(M)
                v = Vector{Float64}(undef, length(v0))
                Π = Vector{Float64}(undef, length(v0))
                Δv = Vector{Float64}(undef, length(v0))
                dΠ = Matrix{Float64}(undef, length(v0), length(v0))
                @testset "Positive and negative definite" begin
                    for _ in 1:3
                        Δv .= ChainRulesTestUtils.rand_tangent(v)
                        v .= v0
                        (vproj, Δvproj) = CRC.frule((nothing, nothing, Δv, nothing), MOD.projection_on_set, d, v, s)
                        @test Δvproj ≈ Δv
                        @test vproj ≈ v
                        v .= -v0
                        dΠ .= MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                        Π .= MOD.projection_on_set(MOD.DefaultDistance(), v, s)
                        (vproj, Δvproj) = CRC.frule((nothing, nothing, Δv, nothing), MOD.projection_on_set, d, v, s)
                        @test dΠ * Δv ≈ Δvproj
                        @test vproj ≈ Π
                    end
                end
            end
        end
    end
    @testset "Indefinite matrix" begin
        s = MOI.PositiveSemidefiniteConeTriangle(2)
        A = Matrix{Float64}(undef, 2, 2)
        Π = Matrix{Float64}(undef, 2, 2)
        v = Vector{Float64}(undef, 3)
        DΠ = Matrix{Float64}(undef, 3, 3)
        Xd = Matrix{Float64}(undef, 2, 2)
        Q = [
            1 0
            0 -1
        ]
        Qi = Q'
        B = [
            0 1/2
            1/2 1
        ]
        for _ in 1:10
            # scale factor
            f = 20 * rand() + 5
            A .= [
                -f 0
                0 f
            ]
            Λ = Diagonal([-f, f])
            Λp = Diagonal([0, f])
            v .= MOD.vec_symm(A)
            vproj = MOD.projection_on_set(MOD.DefaultDistance(), v, s)
            Π .= MOD.unvec_symm(vproj, 2)
            @test Π ≈ Q * Λp * Qi
            DΠ .= MOD.projection_gradient_on_set(d, v, s)
            for _ in 1:20
                Xd .= ChainRulesTestUtils.rand_tangent(Π)
                xd = MOD.vec_symm(Xd)
                dir_deriv_theo = MOD.vec_symm(
                    Q * (B .* (Q' * Xd * Q)) * Q'
                )
                @test DΠ * xd ≈ dir_deriv_theo
                (vproj_frule, Δvproj) = CRC.frule((nothing, nothing, xd, nothing), MOD.projection_on_set, d, v, s)
                @test DΠ * xd ≈ Δvproj
                @test vproj ≈ vproj_frule
            end
        end
    end
end
