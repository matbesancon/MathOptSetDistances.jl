using ChainRulesCore
using ChainRulesTestUtils

import MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
using FiniteDifferences
using Test
using Random

# avoid random finite diff fails because of rounding
Random.seed!(42)

@testset "rrules multivariate" begin
    for n in (1, 2, 10)
        x = randn(n)
        xb = ChainRulesTestUtils.rand_tangent(x)
        s = MOI.Reals(n)
        y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
        yb = ChainRulesTestUtils.rand_tangent(y)
        ChainRulesTestUtils.rrule_test(
            MOD.projection_on_set,
            yb,
            (MOD.DefaultDistance(), nothing),
            (x, xb),
            (s, nothing),
        )
        s = MOI.Zeros(n)
        y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
        yb = ChainRulesTestUtils.rand_tangent(y)
        # requires FillArrays.Zero handling
        # still broken?
        @test_broken ChainRulesTestUtils.rrule_test(
            MOD.projection_on_set,
            yb,
            (MOD.DefaultDistance(), nothing),
            (x, xb),
            (s, nothing),
        )
        for s in (MOI.Nonpositives(n), MOI.Nonnegatives(n))
            y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
            yb = ChainRulesTestUtils.rand_tangent(y)
            ChainRulesTestUtils.rrule_test(
                MOD.projection_on_set,
                yb,
                (MOD.DefaultDistance(), nothing),
                (x, xb),
                (s, nothing),
                atol=10e-5,
            )
        end
        @testset "SOC $n" begin
            s = MOI.SecondOrderCone(n+1)
            for _ in 1:10
                v = 10 * randn(n+1)
                vb = ChainRulesTestUtils.rand_tangent(v)
                if norm(v[2:end]) ≈ v[1]
                    if rand() > 0.8
                        v[1] *= 2
                    else
                        v[1] /= 2
                    end
                end
                y = MOD.projection_on_set(MOD.DefaultDistance(), v, s)
                yb = ChainRulesTestUtils.rand_tangent(y)
                ChainRulesTestUtils.rrule_test(
                    MOD.projection_on_set,
                    yb,
                    (MOD.DefaultDistance(), nothing),
                    (v, vb),
                    (s, nothing),
                    atol=10e-5,
                )
            end
        end
    end
end

@testset "rrules univariate" begin
    @testset "Scalar $ST" for ST in (MOI.EqualTo, MOI.LessThan, MOI.GreaterThan)
        for _ in 1:10
            s = ST(10 * randn())
            x = 10 * randn()
            if isapprox(x, MOI.constant(s), atol=1e-5)
                x *= 2
            end 
            xb = ChainRulesTestUtils.rand_tangent(x)
            y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
            yb = ChainRulesTestUtils.rand_tangent(y)
            sb = ChainRulesTestUtils.rand_tangent(s)
            ChainRulesTestUtils.rrule_test(
                MOD.projection_on_set,
                yb,
                (MOD.DefaultDistance(), nothing),
                (x, xb),
                (s, sb),
                atol=10e-5,
            )
        end
    end
end
