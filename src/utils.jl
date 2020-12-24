
function _bisection(f, left, right; max_iters=10000, tol=1e-10)
    # STOP CODES:
    #   0: Success (floating point limit or exactly 0)
    #   1: Max iters but within tol
    #   2: Failure

    for _ in 1:max_iters
        f_left, f_right = f(left), f(right)
        sign(f_left) == sign(f_right) && error("Interval became non-bracketing.")

        mid = (left + right) / 2
        if left == mid || right == mid
            return mid, 0
        end

        f_mid = f(mid)
        if f_mid == 0
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

    mid = (left + right) / 2
    if abs(f(mid)) < tol
        return mid, 1
    end

    return nothing, 2
end
