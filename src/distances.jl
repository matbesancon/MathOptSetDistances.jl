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
    EpigraphViolationDistance

Distance used when `v ∈ s` can be expressed as a single scalar function `f(v) ≤ 0`.
The distance is expressed as `max(f(v), 0)`.
"""
struct EpigraphViolationDistance <: AbstractDistance end

"""
    distance_to_set(distance_definition, v, s)

Compute the distance of a value to a set.
When `v ∈ s`, the distance is zero (or all individual distances are zero).

Each set `S` implements at least `distance_to_set(d::DefaultDistance, v::T, s::S)`
with `T` of appropriate type for members of the set.
"""
function distance_to_set end

distance_to_set(::AbstractDistance, v, s) = distance_to_set(DefaultDistance(), v, s)
