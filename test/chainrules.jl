using ChainRulesCore
using ChainRulesTestUtils

import MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
using FiniteDifferences
using Test

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
                atol=10e-6,
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

# n = 3
# x = randn(n)
# f = MOD.projection_on_set
# s = MOI.Zeros(n)
# y = MOD.projection_on_set(MOD.DefaultDistance(), x, s)
# yb = ChainRulesTestUtils.rand_tangent(y)
# ȳ = yb
# xb = ChainRulesTestUtils.rand_tangent(x)
# xx̄s=[
#     (MOD.DefaultDistance(), nothing),
#     (x, xb),
#     (s, nothing),
# ]
#
# xs = first.(xx̄s)
# accumulated_x̄ = last.(xx̄s)
# y_ad, pullback = rrule(f, xs...)
# y = f(xs...)
# y_ad == y
#
# ∂s = pullback(ȳ)
# ∂self = ∂s[1]
# x̄s_ad = ∂s[2:end]
# @test ∂self === NO_FIELDS  # No internal fields
# x̄s_is_dne = accumulated_x̄ .== nothing
#
# fdm = FiniteDifferences.central_fdm(5, 1)
#
# x̄s_fd = ChainRulesTestUtils._make_j′vp_call(fdm, (xs...) -> f(xs...), ȳ, xs, x̄s_is_dne)
# f2 = ChainRulesTestUtils._wrap_function(f, xs, x̄s_is_dne)
# ignores = collect(x̄s_is_dne)

@testset "rrules univariate" begin

    @testset "Scalar $ST" for ST in (MOI.EqualTo, MOI.LessThan, MOI.GreaterThan)
        for _ in 1:10
            s = ST(10 * randn())
            x = 10 * randn()
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
            )
        end
    end
end
