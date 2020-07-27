module MathOptSetDistances

import MathOptInterface
using BlockDiagonals
const MOI = MathOptInterface

import LinearAlgebra

export distance_to_set


include("distances.jl")
include("sets.jl")

end # module
