
"""
Represents the standard simplex: `∑x_i ≤ radius, x_i ≥ 0`.
"""
struct StandardSimplex{T} <: MOI.AbstractVectorSet
    dimension::Int
    radius::T
end

"""
Represents the probability simplex: `∑x_i == radius, x_i ≥ 0`.
"""
struct ProbabilitySimplex{T} <: MOI.AbstractVectorSet
    dimension::Int
    radius::T
end

MOI.constant(s::Union{StandardSimplex, ProbabilitySimplex}) = s.radius
MOI.dimension(s::Union{StandardSimplex, ProbabilitySimplex}) = s.dimension

"""
    NormBallFromCone{T, S}

Represents a norm ball built from the corresponding cone with a fixed radius.
"""
struct NormBallFromCone{T, S <: MOI.AbstractSet} <: MOI.AbstractVectorSet
    radius::T
    cone::S
end

const NormOneBall{T} = NormBallFromCone{T, MOI.NormOneCone}
const NormInfinityBall{T} = NormBallFromCone{T, MOI.NormInfinityCone}
const NormTwoBall{T} = NormBallFromCone{T, MOI.SecondOrderCone}
const NormNuclearBall{T} = NormBallFromCone{T, MOI.NormNuclearCone}

MOI.dimension(s::NormBallFromCone) = MOI.dimension(s) - 1
MOI.constant(s::NormBallFromCone) = s.radius
