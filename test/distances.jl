@testset "Set distances" begin
    @testset "$n-dimensional orthants" for n in 1:3:15
        v = rand(n)
        @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Reals(n)) ≈ 0 atol=eps(Float64)
        @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Zeros(n)) > 0
        @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Nonnegatives(n)) ≈ 0 atol=eps(Float64)
        @test MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.Nonpositives(n)) ≈ 0 atol=eps(Float64)
        @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Nonpositives(n)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.Nonnegatives(n)) > 0
    end

    @testset "Scalar comparisons" begin
        values = rand(10)
        for v in values
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.EqualTo(v)) ≈ 0 atol=eps(Float64)
            @test MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.EqualTo(v)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.EqualTo(-v)) ≈ 2v
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.LessThan(v)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.LessThan(v+1)) ≈ 0
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.LessThan(0)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.GreaterThan(0)) ≈ v
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.GreaterThan(v)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), v+1, MOI.GreaterThan(v+1)) ≈ 0
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Interval(v,v)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Interval(-v,v)) ≈ 0
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Interval(-v, 0.0)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.Interval(0.0, v)) ≈ v
        end
    end
    @testset "$n-dimensional norm cones" for n in 2:5:15
        x = rand(n)
        tsum = sum(x)
        vsum = vcat(tsum, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vsum, MOI.NormOneCone(n+1)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), vsum, MOI.NormInfinityCone(n+1)) ≈ 0
        tmax = maximum(x)
        vmax = vcat(tmax, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vmax, MOI.NormOneCone(n+1)) > 0
        @test MOD.distance_to_set(MOD.DefaultDistance(), vmax, MOI.NormInfinityCone(n+1)) ≈ 0 atol=eps(Float64)
        tmin = 0
        vmin = vcat(tmin, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vmin, MOI.NormInfinityCone(n+1)) ≈ tmax
        @test MOD.distance_to_set(MOD.DefaultDistance(), vmin, MOI.NormOneCone(n+1)) ≈ tsum

        tvalid = sqrt(n) # upper bound on the norm2
        vok_soc = vcat(tvalid, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vok_soc, MOI.SecondOrderCone(n+1) ) ≈ 0 atol=eps(Float64)
        vko_soc = vcat(-2, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vko_soc, MOI.SecondOrderCone(n+1) ) ≈ 2 + LinearAlgebra.norm2(x)

        vko_soc = vcat(-2, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vko_soc, MOI.SecondOrderCone(n+1) ) ≈ 2 + LinearAlgebra.norm2(x)

        t_ko_rot = u_ko_rot = LinearAlgebra.norm2(x) / 2
        vko_roc = vcat(t_ko_rot, u_ko_rot, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vko_roc, MOI.RotatedSecondOrderCone(n+2)) ≈ LinearAlgebra.dot(x,x) / 2
        vok_roc = vcat(t_ko_rot * 2, u_ko_rot, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vok_roc, MOI.RotatedSecondOrderCone(n+2)) ≈ 0 atol=10eps(Float64)
    end

    @testset "Geometric Mean cone dimension $n" for n in 2:5:15
        x = rand(n)
        t = 0.5 * prod(x)^(inv(n))
        vok = vcat(t, x)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vok, MOI.GeometricMeanCone(n+1)) ≈ 0 atol=eps(Float64)
        @test MOD.distance_to_set(MOD.DefaultDistance(), vcat(t / 2, x), MOI.GeometricMeanCone(n+1)) ≈ 0 atol=eps(Float64)
        # negative x always means positive distance
        @test MOD.distance_to_set(MOD.DefaultDistance(), vcat(t / 2, vcat(x, -1)), MOI.GeometricMeanCone(n+2)) > 0
        @test MOD.distance_to_set(MOD.DefaultDistance(), vcat(t / 2, -x), MOI.GeometricMeanCone(n+1)) > 0
    end
    
    @testset "Exponential and power cones" begin
        for _ in 1:30
            (x, y, z) = rand(3)
            y += 1 # ensure y > 0
            if y * exp(x/y) <= z
                @test MOD.distance_to_set(MOD.DefaultDistance(), [x, y, z], MOI.ExponentialCone()) ≈ 0 atol=eps(Float64)
                @test MOD.distance_to_set(MOD.DefaultDistance(), [x, -1, z], MOI.ExponentialCone()) ≈ 1 atol=eps(Float64)
            else
                @test MOD.distance_to_set(MOD.DefaultDistance(), [x, y, z], MOI.ExponentialCone()) ≈ y * exp(x/y) - z 
            end
            (u, v, w) = randn(3)
            if u != 0.0 # just in case not to blow up
                if -u*exp(v/u) < ℯ * w && u < 0
                    @test MOD.distance_to_set(MOD.DefaultDistance(), [u, v, w], MOI.DualExponentialCone()) ≈ 0 atol=eps(Float64)
                elseif u < 0
                    @test MOD.distance_to_set(MOD.DefaultDistance(), [u, v, w], MOI.DualExponentialCone()) ≈ -u*exp(v/u) - ℯ * w
                end
                
            end
            (x, y) = randn(2)
            if x < 0 || y < 0
                for e in  (10 * rand(10) .- 5) # e in [-5, 5]
                    @test MOD.distance_to_set(MOD.DefaultDistance(), [x, y, 0.0], MOI.PowerCone(e)) > 0
                end
            else
                for e in  (10 * rand(10) .- 5) # e in [-5, 5]
                    r = x^e * y^(1-e)
                    for z in -r:-0.5:r
                        @test MOD.distance_to_set(MOD.DefaultDistance(), [x, y, z], MOI.PowerCone(e)) ≈ 0 atol=eps(Float64)
                    end
                    @test MOD.distance_to_set(MOD.DefaultDistance(), [x, y, 3r], MOI.PowerCone(e)) ≈ MOD.distance_to_set(MOD.DefaultDistance(), [x, y, -3r], MOI.PowerCone(e)) > 0
                end
            end
    
            (u, v, w) = 10 * rand(3)
            e = rand()
            if 0 < e < 1 # avoid exponents of negatives
                @test MOD.distance_to_set(MOD.DefaultDistance(), [u, v, 0.0], MOI.DualPowerCone(e)) ≈ 0 atol = 10eps(Float64)
                @test MOD.distance_to_set(MOD.DefaultDistance(), [u, v, u^e * v^(1-e) / (e^e * (1-e)^(1-e))], MOI.DualPowerCone(e)) ≈ 0 atol=100eps(Float64)
                @test MOD.distance_to_set(MOD.DefaultDistance(), [u, v, 1 + u^e * v^(1-e) / (e^e * (1-e)^(1-e))], MOI.DualPowerCone(e)) ≈ 1
            end
        end
    end    
end

struct DummyDistance <: MOD.AbstractDistance end

MOD.distance_to_set(::DummyDistance, v, s) = MOD.distance_to_set(MOD.DefaultDistance(), v, s) / 2

@testset "Set non-default distance" begin
    for n in 1:3
        v = rand(n)
        for s in (MOI.Reals(n), MOI.Zeros(n), MOI.Nonnegatives(n), MOI.Nonpositives(n))
            @test MOD.distance_to_set(MOD.DefaultDistance(), v, MOI.Reals(n)) ≈ 2 * MOD.distance_to_set(DummyDistance(), v, MOI.Reals(n)) atol=eps(Float64)
            @test MOD.distance_to_set(MOD.DefaultDistance(), -v, MOI.Reals(n)) ≈ 2 * MOD.distance_to_set(DummyDistance(), -v, MOI.Reals(n)) atol=eps(Float64)
        end
    end
end
