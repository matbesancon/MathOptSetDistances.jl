# MathOptSetDistances

[![Build Status](https://travis-ci.com/matbesancon/MathOptSetDistances.jl.svg?branch=master)](https://travis-ci.com/matbesancon/MathOptSetDistances.jl)
[![Codecov](https://codecov.io/gh/matbesancon/MathOptSetDistances.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/matbesancon/MathOptSetDistances.jl)

Defines the distance of a point `v` to a set `s`. The distance is always 0 if `v ∈ s`.
The API consists of a single function `set_distance(d::D, v::V, s::S)`, with `S` a `MOI.AbstractSet`,
`v` a scalar or vector value and `d` a type of distance.  

New sets should implement at least `set_distance(::DefaultDistance, v::V, s::S)`.
