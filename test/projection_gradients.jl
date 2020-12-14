using Test
using MathOptInterface
const MOI = MathOptInterface

using MathOptSetDistances
const MOD = MathOptSetDistances

using FiniteDifferences
using LinearAlgebra

const bfdm = FiniteDifferences.backward_fdm(5, 1)
const ffdm = FiniteDifferences.forward_fdm(5, 1)

function safe_randn(n)
    v = 2 * randn(n)
    for i in eachindex(v)
        while v[i] ≈ 0
            v[i] = 2 * randn()
        end
    end
    return v
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
            for _ in 1:Ntrials
                L = 3 * tril(rand(n, n))
                M = L * L'
                @testset "Positive definite" begin
                    v = MOD.vec_symm(M)
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    # @test dΠ ≈ grad_fdm
                end
                @testset "Negative definite" begin
                    v = MOD.vec_symm(-M)
                    dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                    grad_fdm1 = FiniteDifferences.jacobian(ffdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    grad_fdm2 = FiniteDifferences.jacobian(bfdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                    @test size(grad_fdm1) == size(grad_fdm2) == size(dΠ)
                    # @test dΠ ≈ grad_fdm
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
                    end
                end
            end
        end
    end
end
