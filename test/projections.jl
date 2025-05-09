using MathOptInterface, SCS, Hypatia

@testset "Test projections distance on vector sets" begin
    for n in [1, 10] # vector sizes
        v = rand(n)
        for s in (MOI.Zeros(n), MOI.Nonnegatives(n), MOI.Nonpositives(n), MOI.Reals(n))
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
    @test MOD.projection_on_set(DD, -ones(5), MOI.Nonpositives(5)) ≈ -ones(5)
    @test MOD.projection_on_set(DD, ones(5), MOI.Nonpositives(5)) ≈ zeros(5)
end

@testset "Trivial projection gradient on vector cones" begin
    #testing POS
    @test MOD.projection_gradient_on_set(DD, -ones(5), MOI.Nonnegatives(5)) ≈ zeros(5,5)
    @test MOD.projection_gradient_on_set(DD, ones(5), MOI.Nonnegatives(5)) ≈  Matrix{Float64}(LinearAlgebra.I, 5, 5)
    @test MOD.projection_gradient_on_set(DD, -ones(5), MOI.Nonpositives(5)) ≈ Matrix{Float64}(LinearAlgebra.I, 5, 5)
    @test MOD.projection_gradient_on_set(DD, ones(5), MOI.Nonpositives(5)) ≈ zeros(5,5)

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
    c1 = MOI.PositiveSemidefiniteConeTriangle(4)
    output_1 = MOD.projection_on_set(DD, v1, c1)
    output_2 = MOD.projection_on_set(DD, v2, c2)
    output_joint = MOD.projection_on_set(DD, [v1, v2], [c1, c2])
    @test output_joint ≈ [output_1; output_2]
end

@testset "Non-trivial block projection gradient" begin
    v1 = rand(Float64, 15)
    v2 = rand(Float64, 5)
    c1 = MOI.PositiveSemidefiniteConeTriangle(5)
    c2 = MOI.SecondOrderCone(5)

    output_1 = MOD.projection_gradient_on_set(DD, v1, c1)
    output_2 = MOD.projection_gradient_on_set(DD, v2, c2)
    output_joint = MOD.projection_gradient_on_set(DD, [v1, v2], [c1, c2])
    @test output_joint ≈ BlockDiagonal([output_1, output_2])
end

@testset "Scaled projection" begin
    a = [1, 1.0, 1]
    b = [1, sqrt(2), 1]
    set = MOI.PositiveSemidefiniteConeTriangle(2)
    @test MOD.projection_on_set(DD, a, set) ≈ a
    @test MOD.projection_on_set(DD, b, set) ≈ 1.20710678 * a
    @test MOD.projection_on_set(DD, a, MOI.Scaled(set)) ≈ a
    @test MOD.projection_on_set(DD, b, MOI.Scaled(set)) ≈ b
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
        px = MOD.projection_on_set(DD, x, cone)

        # Tests for big numbers -- use optimality conditions
        if maximum(abs.(x)) > exp(9)
            proj = dual ? px - x : px
            pdx = dual ? -x - proj : x - proj
            ortho = abs(dot(proj, pdx)) / norm(x)
            if ortho >= tol || !MOD._in_exp_cone(proj, tol=tol)
                error("x = $x,\npx = $px,\npdx = $pdx,\northogonality: $ortho")
            end
            return true
        end

        model = MOI.instantiate(SCS.Optimizer, with_bridge_type = Float64)
        MOI.set(model, MOI.Silent(), true)

        MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1e-7)
        MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), 1e-7)
        MOI.set(model, MOI.RawOptimizerAttribute("max_iters"), 10_000)

        z = MOI.add_variables(model, 3)
        t = MOI.add_variable(model)

        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            1.0 * t,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        MOI.add_constraint(
            model,
            MOI.VectorOfVariables(z),
            cone,
        )

        MOI.Utilities.normalize_and_add_constraint(
            model,
            sum((1.0 * z[i] - x[i])^2 for i in 1:3) - t,
            MOI.LessThan(0.0),
        )

        MOI.optimize!(model)
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL

        z_star = MOI.get.(model, MOI.VariablePrimal(), z)

        if !isapprox(px, z_star, atol=tol)
            error("Exp cone projection failed:\n x = $x\nMOD: $px\nMOI-SCS: $z_star
                   norm: $(norm(px - z_star))")
            return false
       end
       return true
    end

    Random.seed!(0)
    tol = 1e-4
    case_p = zeros(4)
    case_d = zeros(4)
    exponents = [10, 20]
    domain = [-exp.(exponents); 0.0; exp.(exponents)]
    x = randn(3)
    @inferred MOD.projection_on_set(DD, x, MOI.ExponentialCone())
    @inferred MOD.projection_on_set(DD, x, MOI.DualExponentialCone())
    for (x1, x2, x3) in Iterators.product(domain, domain, domain)
        # x = randn(3)
        x = [x1, x2, x3]

        case_p[det_case_exp_cone(x; dual=false)] += 1
        @test _test_proj_exp_cone_help(x, tol; dual=false)

        case_d[det_case_exp_cone(x; dual=true)] += 1
        @test _test_proj_exp_cone_help(x, tol; dual=true)

        x = randn(3)
        case_p[det_case_exp_cone(x; dual=false)] += 1
        @test _test_proj_exp_cone_help(x, tol; dual=false)

        case_d[det_case_exp_cone(x; dual=true)] += 1
        @test _test_proj_exp_cone_help(x, tol; dual=true)
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
        model = MOI.instantiate(Hypatia.Optimizer, with_bridge_type = Float64)
        # model = MOI.instantiate(SCS.Optimizer, with_bridge_type = Float64)
        MOI.set(model, MOI.Silent(), true)

        # MOI.set(model, MOI.RawOptimizerAttribute("eps_abs"), 1e-10)
        # MOI.set(model, MOI.RawOptimizerAttribute("eps_rel"), 1e-10)
        # MOI.set(model, MOI.RawOptimizerAttribute("max_iters"), 10_000)

        z = MOI.add_variables(model, 3)
        t = MOI.add_variable(model)

        MOI.set(
            model,
            MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            1.0 * t,
        )
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)

        MOI.add_constraint(
            model,
            MOI.VectorOfVariables(z),
            cone,
        )

        MOI.Utilities.normalize_and_add_constraint(
            model,
            sum((1.0 * z[i] - x[i])^2 for i in 1:3) - t,
            MOI.LessThan(0.0),
        )

        MOI.optimize!(model)

        z_star = MOI.get.(model, MOI.VariablePrimal(), z)
        px = MOD.projection_on_set(DD, x, cone)
        if !isapprox(px, z_star, atol=tol)
            error("x = $x\nα = $α\nnorm = $(norm(px - z_star))\npx=$px\ntrue=$z_star")
            return false
       end
       if !MOD._in_pow_cone(px, cone, tol=tol)
           error("Not in power cone\nx = $x\nα = $α\nnorm = $(norm(px - z_star))\npx=$px\ntrue=$z_star")
           return false
       end
       return true
    end

    Random.seed!(0)
    n = 3
    atol = 5e-6
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

    @testset "Difficult Projections" begin
        # these are difficult projections that typically fail with Newton method
        candidates = [
            ([-2.1437051121224258, 1.9673722101310502, 0.8499144338875969], 0.15),
            ([1.2782001669773202, -0.024746244349335353, 0.020923505930442263], 0.50),
            ([1.635118813047526, -1.7337027657509976, 0.1521032906785757], 0.85),
            ([-1.0626658716421513, 0.2640095391880903, -0.18047124406952353], 0.05),
            ([0.7154127036250458, -1.2726229049439148, -0.20181164946348515], 0.95),
        ]
        for (x, α) in candidates
            @test _test_proj_pow_cone_help(x, α, atol; dual=false)
        end
    end
end

@testset "Simplex projections" begin
    for n in (1, 2, 10)
        for _ in 1:10
            s = MOD.StandardSimplex(n, rand())
            sp = MOD.ProbabilitySimplex(n, rand())
            for _ in 1:5
                v = 10 * randn(n)
                p = MOD.projection_on_set(DD, v, s)
                @test all(p .>= 0)
                @test any(v .> 0) || sum(p) ≈ 0
                if sum(p) > 0
                    @test sum(p) <= s.radius + 10 * n * eps()
                end
                p = MOD.projection_on_set(DD, v, sp)
                @test all(p .>= 0)
                @test sum(p) ≈ sp.radius
                vu = rand(n)
                vu ./= sum(vu)
                vu .*= 0.9 * s.radius
                @test vu ≈ MOD.projection_on_set(DD, vu, s)
                vu ./= sum(vu)
                vu .*= 0.9 * sp.radius
                @test !≈(vu, MOD.projection_on_set(DD, vu, sp))
                vu ./= sum(vu)
                vu .*= sp.radius 
                @test ≈(vu, MOD.projection_on_set(DD, vu, sp))
            end
        end
    end
end
