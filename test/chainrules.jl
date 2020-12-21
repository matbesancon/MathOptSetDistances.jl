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
