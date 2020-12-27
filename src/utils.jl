
function _bisection(f, left, right; max_iters=500, tol=1e-14)
    # STOP CODES:
    #   0: Success (floating point limit or exactly 0)
    #   1: Failure (max_iters without coming within tolerance of 0)

    for _ in 1:max_iters
        f_left, f_right = f(left), f(right)
        sign(f_left) == sign(f_right) && error("Interval became non-bracketing.")

        # Terminate if interval length ~ floating point precision (< eps())
        mid = (left + right) / 2
        if left == mid || right == mid
            return mid, 0
        end

        # Terminate if within tol of 0; otherwise, bisect
        f_mid = f(mid)
        if abs(f_mid) < tol
            return mid, 0
        end
        if sign(f_mid) == sign(f_left)
            left = mid
            continue
        end
        if  sign(f_mid) == sign(f_right)
            right = mid
            continue
        end
    end

    return nothing, 1
end
