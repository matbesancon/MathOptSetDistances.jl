const DD = MOD.DefaultDistance()

@testset "Test projections distance on vector sets" begin
    for n in [1, 10] # vector sizes
        v = rand(n)
        for s in (MOI.Zeros(n), MOI.Nonnegatives(n), MOI.Reals(n))
            πv = MOD.projection_on_set(DD, v, s)
            @test MOD.distance_to_set(DD, πv, s) ≈ 0 atol=eps(Float64)
        end
    end
end

@testset "Test projection distance on scalar sets" begin
    v = rand()
    for s in (MOI.EqualTo(v), )
        πv = MOD.projection_on_set(DD, v, s)
        @test MOD.distance_to_set(DD, πv, s) ≈ 0 atol=eps(Float64)
    end
end

@testset "Trivial projection on vector cones" begin
    #testing POS
    @test MOD.projection_on_set(DD, -ones(5), MOI.Nonnegatives(5)) ≈ zeros(5)
    @test MOD.projection_on_set(DD, ones(5), MOI.Nonnegatives(5)) ≈ ones(5)
end

@testset "Trivial projection gradient on vector cones" begin
    #testing POS
    @test MOD.projection_gradient_on_set(DD, -ones(5), MOI.Nonnegatives(5)) ≈ zeros(5,5)
    @test MOD.projection_gradient_on_set(DD, ones(5), MOI.Nonnegatives(5)) ≈  Matrix{Float64}(LinearAlgebra.I, 5, 5)

    # testing SOC
    @test MOD.projection_gradient_on_set(DD, [1.0; ones(1)], MOI.SecondOrderCone(1)) ≈ [1.0  0.0;
                                                                                        0.0  1.0]
    @test MOD.projection_gradient_on_set(DD, [0.0; ones(4)], MOI.SecondOrderCone(4)) ≈ [0.5   0.25  0.25  0.25  0.25;
                                                                                        0.25  0.5   0.0   0.0   0.0;
                                                                                        0.25  0.0   0.5   0.0   0.0;
                                                                                        0.25  0.0   0.0   0.5   0.0;
                                                                                        0.25  0.0   0.0   0.0   0.5]

    # testing SDP trivial
    # eye4 = Matrix{Float64}(LinearAlgebra.I, 4, 4)
    eye4 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]  # symmetrical PSD triangle format
    eye10 = Matrix{Float64}(LinearAlgebra.I, 10, 10)
    @test MOD.projection_gradient_on_set(DD, eye4, MOI.PositiveSemidefiniteConeTriangle(4)) ≈ eye10
end

@testset "Non-trivial joint projection" begin
    v1 = rand(Float64, 10)
    v2 = rand(Float64, 5)
    c1 = MOI.PositiveSemidefiniteConeTriangle(5)
    c2 = MOI.SecondOrderCone(5)

    output_1 = MOD.projection_on_set(DD, v1, c1)
    output_2 = MOD.projection_on_set(DD, v2, c2)
    output_joint = MOD.projection_on_set(DD, [v1, v2], [c1, c2])
    @test output_joint' ≈ [output_1' output_2']
end

@testset "Non-trivial Block projection gradient" begin
    v1 = rand(Float64, 15)
    v2 = rand(Float64, 5)
    c1 = MOI.PositiveSemidefiniteConeTriangle(6)
    c2 = MOI.SecondOrderCone(5)

    output_1 = MOD.projection_gradient_on_set(DD, v1, c1)
    output_2 = MOD.projection_gradient_on_set(DD, v2, c2)
    output_joint = MOD.projection_gradient_on_set(DD, [v1, v2], [c1, c2])
    @test output_joint ≈ BlockDiagonal([output_1, output_2])
end
