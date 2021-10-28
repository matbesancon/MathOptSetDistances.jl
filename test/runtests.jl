
using Test

import MathOptSetDistances
const MOD = MathOptSetDistances
const MOI = MathOptSetDistances.MOI
const DD = MOD.DefaultDistance()

using LinearAlgebra
using Random
import BlockDiagonals: BlockDiagonal

using FiniteDifferences

const bfdm = FiniteDifferences.backward_fdm(5, 1)
const ffdm = FiniteDifferences.forward_fdm(5, 1)

include("distances.jl")
include("projections.jl")
include("projection_gradients.jl")
include("chainrules.jl")
