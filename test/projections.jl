using JuMP, SCS
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
    v = 5 * randn()
    for s in (MOI.EqualTo(v), MOI.LessThan(v), MOI.GreaterThan(v))
        @test MOD.distance_to_set(DD, v, s) ≈ 0 atol=eps(Float64)
        @test MOD.projection_on_set(DD, v, s) ≈ v
        vm = -v
        πvm = MOD.projection_on_set(DD, vm, s)
        if vm < 0 && !isa(s, MOI.LessThan)
            @test MOD.distance_to_set(DD, vm, s) > 0
            @test abs(vm - πvm) ≈ MOD.distance_to_set(DD, vm, s)
        elseif vm > 0 && !isa(s, MOI.GreaterThan)
            @test MOD.distance_to_set(DD, vm, s) > 0
            @test abs(vm - πvm) ≈ MOD.distance_to_set(DD, vm, s)
        end
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
    c2 = MOI.SecondOrderCone(5)
    v1 = rand(Float64, 10)
    v2 = rand(Float64, 5)
    c1 = MOI.PositiveSemidefiniteConeTriangle(5)
    output_1 = MOD.projection_on_set(DD, v1, c1)
    output_2 = MOD.projection_on_set(DD, v2, c2)
    output_joint = MOD.projection_on_set(DD, [v1, v2], [c1, c2])
    @test output_joint ≈ [output_1; output_2]
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


@testset "Exponential Cone Projections" begin
    function det_case_exp_cone(v; dual=false)
        v = dual ? -v : v
        if MOD.distance_to_set(DD, v, MOI.ExponentialCone()) < 1e-8
            return 1
        elseif MOD.distance_to_set(DD, -v, MOI.DualExponentialCone()) < 1e-8
            return 2
        elseif v[1] <= 0 && v[2] <= 0 #TODO: threshold here??
            return 3
        else
            return 4
        end
    end

    function _test_proj_exp_cone_help(x, tol; dual=false)
        cone = dual ? MOI.DualExponentialCone() : MOI.ExponentialCone()
        model = Model()
        set_optimizer(model, optimizer_with_attributes(
            SCS.Optimizer, "eps" => 1e-10, "max_iters" => 10000, "verbose" => 0))
        @variable(model, z[1:3])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in cone)
        optimize!(model)
        z_star = value.(z)
        px = MOD.projection_on_set(DD, x, cone)
        if !isapprox(px, z_star, atol=tol)
            # error("Exp cone projection failed:\n x = $x\nMOD: $px\nJuMP: $z_star
            #        \nnorm: $(norm(px - z_star))")
            return false
       end
       return true
    end

    Random.seed!(0)
    n = 3
    atol = 1e-7
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        x = randn(3)

        case_p[det_case_exp_cone(x; dual=false)] += 1
        @test _test_proj_exp_cone_help(x, atol; dual=false)

        case_d[det_case_exp_cone(x; dual=true)] += 1
        @test _test_proj_exp_cone_help(x, atol; dual=true)
    end
    @test all(case_p .> 0) && all(case_d .> 0)
end

@testset "Power Cone Projections" begin
    function det_case_pow_cone(x, α; dual=false)
        v = dual ? -x : x
        s = MOI.PowerCone(α)
        if MOD._in_pow_cone(v, s)
            return 1
        elseif MOD._in_pow_cone(v, MOI.dual_set(s))
            return 2
        elseif abs(v[3]) <= 1e-8
            return 3
        else
            return 4
        end
    end

    function _test_proj_pow_cone_help(x, α, tol; dual=false)
        cone = dual ? MOI.DualPowerCone(α) : MOI.PowerCone(α)
        model = Model()
        set_optimizer(model, optimizer_with_attributes(
            SCS.Optimizer, "eps" => 1e-10, "max_iters" => 10000, "verbose" => 0))
        @variable(model, z[1:3])
        @variable(model, t)
        @objective(model, Min, t)
        @constraint(model, sum((x-z).^2) <= t)
        @constraint(model, z in cone)
        optimize!(model)
        z_star = value.(z)
        px = MOD.projection_on_set(DD, x, cone)
        if !isapprox(px, z_star, atol=tol)
            # error("x = $x\nα = $α\nnorm = $(norm(px - z_star))\npx=$px\ntrue=$z_star")
            return false
       end
       return true
    end

    Random.seed!(0)
    n = 3
    atol = 2e-7
    case_p = zeros(4)
    case_d = zeros(4)
    for _ in 1:100
        x = randn(3)
        for α in [rand(0.05:0.05:0.95); 0.5]

            # Need to get some into case 3
            if rand(1:10) == 1
                x[3] = 0
            end

            case_p[det_case_pow_cone(x, α; dual=false)] += 1
            @test _test_proj_pow_cone_help(x, α, atol; dual=false)

            case_d[det_case_pow_cone(x, α; dual=true)] += 1
            @test _test_proj_pow_cone_help(x, α, atol; dual=true)
        end
    end
    @test all(case_p .> 0) && all(case_d .> 0)
end
