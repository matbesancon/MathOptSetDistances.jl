module MathOptSetDistances

import MathOptInterface
const MOI = MathOptInterface
using BlockDiagonals: BlockDiagonal
import FillArrays
using LinearAlgebra
import ChainRulesCore
import FiniteDifferences

export distance_to_set, projection_on_set, projection_gradient_on_set

include("distances.jl")
include("distance_sets.jl")
include("projections.jl")
include("chainrules.jl")

end # module
