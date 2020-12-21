
function FiniteDifferences.to_vec(s::S) where {S <: Union{MOI.EqualTo, MOI.LessThan, MOI.GreaterThan}}
    function set_from_vec(v)
        return S(v[1])
    end
    return [MOI.constant(s)], set_from_vec
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::T, s::MOI.EqualTo) where {T}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), ChainRulesCore.Zero(), ChainRulesCore.Composite{typeof(s)}(value=Δvproj))
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::T, s::MOI.LessThan) where {T}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        if vproj == v # if exactly equal, then constraint inactive
            return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), Δvproj, ChainRulesCore.Composite{typeof(s)}(upper=zero(Δvproj)))
        end
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), zero(Δvproj), ChainRulesCore.Composite{typeof(s)}(upper=Δvproj))
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::T, s::MOI.GreaterThan) where {T}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        if vproj == v # if exactly equal, then constraint inactive
            return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), Δvproj, ChainRulesCore.Composite{typeof(s)}(lower=zero(Δvproj)))
        end
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), zero(Δvproj), ChainRulesCore.Composite{typeof(s)}(lower=Δvproj))
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::AbstractVector{T}, s::MOI.Reals) where {T}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), Δvproj, ChainRulesCore.DoesNotExist())
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::AbstractVector{T}, s::MOI.Zeros) where {T}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), FillArrays.Zeros(length(v)), ChainRulesCore.DoesNotExist())
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::DefaultDistance, v::AbstractVector{T}, s::S) where {T, S <: Union{MOI.Nonnegatives, MOI.Nonpositives}}
    vproj = projection_on_set(d, v, s)
    function pullback(Δvproj)
        ∂v = zeros(eltype(Δvproj), length(Δvproj))
        for i in eachindex(Δvproj)
            if vproj[i] == v[i]
                ∂v[i] = Δvproj[i]
            end
        end
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), ∂v, ChainRulesCore.DoesNotExist())
    end
    return (vproj, pullback)
end

function ChainRulesCore.rrule(::typeof(projection_on_set), d::Union{DefaultDistance, NormedEpigraphDistance}, v::AbstractVector{T}, s::MOI.SecondOrderCone) where {T}
    vproj = projection_on_set(d, v, s)
    t = v[1]
    x = v[2:end]
    norm_x = LinearAlgebra.norm2(x)
    function pullback(Δv)
        v̄ = zeros(eltype(Δv), length(Δv))
        if norm_x ≤ t
            v̄ .= Δvproj
            return v̄
        elseif norm_x ≤ -t
            return v̄
        end
        v̄[1] = inv(2norm_x) * sum(Δv[i] * v[i] for i in 2:length(v)) + Δv[1] / 2
        inv_norm = inv(2norm_x)
        dot_prod = LinearAlgebra.dot(x, Δv[2:end])
        v̄[2:length(v)] .= inv_norm * x * Δv[1]
        v̄[2:length(v)] .+= inv_norm * (t + norm_x) * Δv[2:end]
        v̄[2:length(v)] .-= inv_norm^2 * t * x * dot_prod
        return (ChainRulesCore.NO_FIELDS, ChainRulesCore.DoesNotExist(), v̄, ChainRulesCore.DoesNotExist())
    end
    return (vproj, pullback)
end
