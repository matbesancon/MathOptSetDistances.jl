using Test
using MathOptInterface
const MOI = MathOptInterface

using MathOptSetDistances
const MOD = MathOptSetDistances

using FiniteDifferences

const fdm = FiniteDifferences.central_fdm(5, 1)

@testset "Test gradients with finite differences" begin
    Ntrials = 5
    @testset "Dimension $n" for n in (1, 3, 10)
        vector_sets = (
            MOI.Zeros(n),
            MOI.Reals(n),
            MOI.Nonnegatives(n),
            MOI.Nonpositives(n),
        )
        vs = [2 * randn(n) for _ in 1:Ntrials]
        for _ in 1:Ntrials
            push!(vs, rand(-5:5, n))
        end
        @testset "Vector set $s" for s in vector_sets
            for v in vs
                dΠ = MOD.projection_gradient_on_set(MOD.DefaultDistance(), v, s)
                grad_fdm = FiniteDifferences.jacobian(fdm, x -> MOD.projection_on_set(MOD.DefaultDistance(), x, s), v)[1]'
                @test size(grad_fdm) == size(dΠ)
                @test dΠ ≈ grad_fdm
            end
        end
    end
end
