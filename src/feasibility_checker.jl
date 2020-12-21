const MOD = MathOptSetDistances
const MOIU = MOI.Utilities

variable_primal(model) = x -> MOI.get(model, MOI.VariablePrimal(), x)
variable_primal_start(model) = x -> MOI.get(model, MOI.VariablePrimalStart(), x)

mutable struct FeasibilityCheckerOptions
    names::Bool
    index::Bool
    function FeasibilityCheckerOptions(;names = true, index = true)
        return new(
            names,
            index,
        )
    end
end

function constraint_violation_report(model::MOI.ModelLike;
    varval::Function = variable_primal(model),
    distance_map::AbstractDict = Dict(),
    options = FeasibilityCheckerOptions())

    largest, vec = constraint_violation(model, varval = varval, distance_map = distance_map)

    str = " Feasibility report\n\n"

    str *= " Maximum overall violation = $(largest)\n\n"

    str *= " Maximum violation per constraint type\n"

    # sort!(vec)

    for (val, ref) in vec
        str *= violation_string(model, ref, val, distance_map, options)
    end

    return str
end

function violation_string(model, ref::MOI.ConstraintIndex{F, S}, val, distance_map, options) where {F, S}
    str = " ($F, $S) = $(val)"
    if haskey(distance_map, (F, S))
        str *= " [$(distance_map[F, S])]"
    end
    str *= "\n"
    if options.index
        str *= "     Index: $(ref)\n"
    end
    if options.names && MOI.supports(model, MOI.ConstraintName(), MOI.ConstraintIndex{F, S})
        name = try
            MOI.get(model, MOI.ConstraintName(), ref)
        catch
            nothing
        end
        if name === nothing 
            str *= "     Name: $(name)\n"
        end
    end
    return str
end

function constraint_violation(model::MOI.ModelLike;
    varval::Function = variable_primal(model),
    distance_map::Dict = Dict())
    vec = Any[]
    largest = 0.0
    for (F, S) in MOI.get(model, MOI.ListOfConstraints())
        distance = if haskey(distance_map, (F, S))
            distance_map[F, S]
        else
            MOD.DefaultDistance()
        end
        val, ref = constraint_violation(model, F, S, varval = varval, distance = distance)
        push!(vec, (val, ref))
        if val >= largest
            largest = val
        end
    end
    return largest, vec
end

function constraint_violation(model::MOI.ModelLike,
    ::Type{F}, ::Type{S};
    varval::Function = variable_primal(model),
    distance::MOD.AbstractDistance = MOD.DefaultDistance()) where {F, S}
    largest = 0.0
    largest_ref = MOI.ConstraintIndex{F, S}(-1)
    list = MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
    if isempty(list)
        return NaN, nothing
    end
    for con in list
        val = constraint_violation(model, con, varval = varval, distance = distance)
        if val >= largest
            largest = val
            largest_ref = con
        end
    end
    return largest, largest_ref
end

function constraint_violation(model::MOI.ModelLike, con::MOI.ConstraintIndex;
    varval::Function = variable_primal(model),
    distance::MOD.AbstractDistance = MOD.DefaultDistance())
    func = MOI.get(model, MOI.ConstraintFunction(), con)
    set  = MOI.get(model, MOI.ConstraintSet(), con)
    val  = MOIU.eval_variables(varval, func)
    dist = distance_to_set(distance, val, set)
    return dist
end
