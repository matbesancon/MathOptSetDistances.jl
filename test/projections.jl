@testset "Test projections distance on vector sets" begin
    for n in [1, 10] # vector sizes
        v = rand(n)
        for s in (MOI.Zeros(n), MOI.Nonnegatives(n))
            πv = MOD.projection_on_set(MOD.DefaultDistance(), s, v)
            @test MOD.distance_to_set(MOD.DefaultDistance(), πv, s) ≈ 0 atol=eps(Float64)
        end
    end
end

@testset "Test projection distance on scalar sets" begin
    v = rand()
    for s in (MOI.EqualTo(v), )
        πv = MOD.projection_on_set(MOD.DefaultDistance(), s, v)
        @test MOD.distance_to_set(MOD.DefaultDistance(), πv, s) ≈ 0 atol=eps(Float64)
    end
end
