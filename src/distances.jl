"""
    AbstractDistance

Distance function used to evaluate the distance from a point to a set.
New subtypes of `AbstractDistance` must implement fallbacks for sets they don't cover and implement
`distance_to_set(::Distance, v, s::S)` for sets they override the distance for.
"""
abstract type AbstractDistance end

"""
    DefaultDistance

Default distance function, implemented for all sets.
"""
struct DefaultDistance <: AbstractDistance end

"""
    distance_to_set(distance_definition, v, s)

Compute the distance of a value to a set.
When `v ∈ s`, the distance is zero (or all individual distances are zero).

Each set `S` implements at least `distance_to_set(d::DefaultDistance, v::T, s::S)`
with `T` of appropriate type for members of the set.
"""
function distance_to_set end

distance_to_set(::AbstractDistance, v, s) = distance_to_set(DefaultDistance(), v, s)

"""
    EpigraphViolationDistance

Distance used when `v ∈ s` can be expressed as a single scalar function `f(v) ≤ 0`.
The distance is expressed as `max(f(v), 0)`.
"""
struct EpigraphViolationDistance <: AbstractDistance end

"""
    NormedEpigraphDistance{p}

Distance used when `v ∈ s` can be expressed as a several functions `f_i(v) ≤ 0 ∀ i ∈ I`.
The distance is expressed as `norm([max(f_i(v), 0) ∀ i ∈ I], n)` with `p` a norm value (2, 1, Inf).
"""
struct NormedEpigraphDistance{p} <: AbstractDistance end

# _norm_set(::NormedEpigraphDistance{1}) = MOI.NormOneCone
# _norm_set(::NormedEpigraphDistance{2}) = MOI.SecondOrderCone
# _norm_set(::NormedEpigraphDistance{Inf}) = MOI.NormInfCone
