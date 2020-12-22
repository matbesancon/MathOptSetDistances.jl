
using Test

using MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
const MOIU = MOI.Utilities

using LinearAlgebra
import BlockDiagonals: BlockDiagonal

include("distances.jl")
include("projections.jl")
include("feasibility_checker.jl")
include("projection_gradients.jl")
include("chainrules.jl")
