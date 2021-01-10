
"""
    _bisection(f, left::T, right::T; max_iters=500, tol=1e-14) -> Union{Nothing, T}

Finds the root `x, f(x) ≈ 0` in interval `[left, right]`, returns nothing if convergence fails.
Errors if the interval is non-bracketing (sign of `f(left)` identical to sign of `f(right)`).
"""
function _bisection(f, left, right; max_iters=500, tol=1e-14)
    for _ in 1:max_iters
        f_left, f_right = f(left), f(right)
        signbit(f_left) == signbit(f_right) && error("Interval became non-bracketing.")

        # Terminate if interval length ~ eps()
        mid = (left + right) / 2
        if left == mid || right == mid
            return mid
        end
        # Terminate if within tol of 0; otherwise, bisect
        f_mid = f(mid)
        if abs(f_mid) < tol
            return mid
        end
        if signbit(f_mid) == signbit(f_left)
            left = mid
        elseif signbit(f_mid) == signbit(f_right)
            right = mid
        end
    end
    return nothing
end
