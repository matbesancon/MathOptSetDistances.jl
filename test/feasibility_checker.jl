@testset "Feasibility checker" begin

@testset "From primal start simple" begin
    mock = MOIU.MockOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()))
    # mock = MOIU.MockOptimizer(MOIU.Model{Float64}())

    MOIU.loadfromstring!(mock,
    """
        variables: x
        minobjective: 2.0x
        c1: x >= 1.0
        c2: x <= 2.0
    """)

    x = MOI.get(mock, MOI.VariableIndex, "x")

    for val in [0.5, 1.0, 1.5, 2.0, 2.5]
        MOI.set(mock, MOI.VariablePrimalStart(), x, val)
        @test MOI.get(mock, MOI.VariablePrimalStart(), x) == val

        largest, list = MOD.constraint_violation(mock, varval = MOD.variable_primal_start(mock))
        @test length(list) == 2
        @test largest == abs(val - clamp(val, 1.0, 2.0))

        ref = MOI.get(mock, MOI.ListOfConstraintIndices{
            MOI.SingleVariable, MOI.LessThan{Float64}}())[1]
        largest = MOD.constraint_violation(mock, ref, varval = MOD.variable_primal_start(mock))
        @test largest == abs(val - min(val, 2.0))

        largest, ref = MOD.constraint_violation(mock,
            MOI.SingleVariable, MOI.LessThan{Float64},
            varval = MOD.variable_primal_start(mock))
        @test largest == abs(val - min(val, 2.0))

        ref = MOI.get(mock, MOI.ListOfConstraintIndices{
            MOI.SingleVariable, MOI.GreaterThan{Float64}}())[1]
        largest = MOD.constraint_violation(mock, ref, varval = MOD.variable_primal_start(mock))
        @test largest == abs(val - max(val, 1.0))

        largest, ref = MOD.constraint_violation(mock,
            MOI.SingleVariable, MOI.GreaterThan{Float64},
            varval = MOD.variable_primal_start(mock))
        @test largest == abs(val - max(val, 1.0))

        largest, ref = MOD.constraint_violation(mock,
            MOI.SingleVariable, MOI.EqualTo{Float64},
            varval = MOD.variable_primal_start(mock))
        @test isnan(largest)
        @test ref === nothing

    end

    str = MOD.constraint_violation_report(mock, varval = MOD.variable_primal_start(mock))

    println(str)
end

@testset "From primal start sum" begin
    mock = MOIU.MockOptimizer(MOIU.UniversalFallback(MOIU.Model{Float64}()))
    # mock = MOIU.MockOptimizer(MOIU.Model{Float64}())

    MOIU.loadfromstring!(mock,
    """
        variables: x, y
        minobjective: 2.0x
        c1: x + 1*y >= 1.0
        c2: x + (-1)*y <= 2.0
    """)

    x = MOI.get(mock, MOI.VariableIndex, "x")
    y = MOI.get(mock, MOI.VariableIndex, "y")

    MOI.set(mock, MOI.VariablePrimalStart(), x, 2.0)
    @test MOI.get(mock, MOI.VariablePrimalStart(), x) == 2.0

    MOI.set(mock, MOI.VariablePrimalStart(), y, -2.0)
    @test MOI.get(mock, MOI.VariablePrimalStart(), y) == -2.0

    largest, ref = MOD.constraint_violation(mock,
        MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64},
        varval = MOD.variable_primal_start(mock))
    @test largest == 2.0

    largest, ref = MOD.constraint_violation(mock,
        MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64},
        varval = MOD.variable_primal_start(mock))
    @test largest == 1.0


end


end