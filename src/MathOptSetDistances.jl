module MathOptSetDistances

import MathOptInterface
const MOI = MathOptInterface
using BlockDiagonals: BlockDiagonal
import FillArrays
using LinearAlgebra

using StaticArrays: @SMatrix, SMatrix, @SVector, SVector

export distance_to_set, projection_on_set, projection_gradient_on_set

include("sets.jl")
include("utils.jl")
include("distances.jl")
include("distance_sets.jl")
include("projections.jl")
include("chainrules.jl")

end # module
