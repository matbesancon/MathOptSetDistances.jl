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

@testset "rrules multivariate" begin
    for n in (1, 2, 10)
        x = randn(n)
        s = MOI.Reals(n)
        y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
        dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), x, s)
        for _ in 1:10
            xb = ChainRulesTestUtils.rand_tangent(x)
            yb = ChainRulesTestUtils.rand_tangent(y)
            ChainRulesTestUtils.rrule_test(
                MOD.projection_on_set,
                yb,
                (MOD.DefaultDistance(), nothing),
                (x, xb),
                (s, nothing),
            )
            (yprimal, pullback) = CRC.rrule(MOD.projection_on_set, MOD.DefaultDistance(), x, s)
            @test yprimal ≈ y
            (_, _, Δx, _) = pullback(yb)
            @test Δx ≈ dΠ' * yb
        end
        s = MOI.Zeros(n)
        y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
        yb = ChainRulesTestUtils.rand_tangent(y)
        xb = ChainRulesTestUtils.rand_tangent(x)
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
            xb = ChainRulesTestUtils.rand_tangent(x)
            ChainRulesTestUtils.rrule_test(
                MOD.projection_on_set,
                yb,
                (MOD.DefaultDistance(), nothing),
                (x, xb),
                (s, nothing),
                atol=1e-4,
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
                    atol=1e-4,
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
                atol=1e-4,
            )
        end
    end
end
