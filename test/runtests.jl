
using Test

using MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
const MOIU = MOI.Utilities

using Test
import LinearAlgebra
import BlockDiagonals: BlockDiagonal

include("distances.jl")

include("projections.jl")

include("feasibility_checker.jl")
