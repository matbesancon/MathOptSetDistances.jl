
using Test

using MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI

using Test
using LinearAlgebra
using JuMP, SCS
using Random
import BlockDiagonals: BlockDiagonal

include("distances.jl")
include("projections.jl")
include("projection_gradients.jl")
