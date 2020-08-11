module MathOptSetDistances

import MathOptInterface
using BlockDiagonals: BlockDiagonal
const MOI = MathOptInterface

import LinearAlgebra

export distance_to_set, projection_on_set, projection_gradient_on_set


include("distances.jl")
include("distance_sets.jl")
include("projections.jl")

end # module
