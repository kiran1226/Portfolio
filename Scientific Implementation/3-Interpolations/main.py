import numpy as np


####################################################################################################
# Exercise 1: Interpolation

def lagrange_interpolation(x: np.ndarray, y: np.ndarray) -> (np.poly1d, list):
    """
    Generate Lagrange interpolation polynomial.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    polynomial: polynomial as np.poly1d object
    base_functions: list of base polynomials
    """

    assert (x.size == y.size)
    polynomial = np.poly1d(0)

    base_functions = []
    base_functions1 = []
    x2 = np.poly1d([1, 0])
    y2 = np.poly1d([1])

    for i in range(x.size):
        for k in range(x.size):
            if k != i:
                c = (x2 - x[k])
                d = (x[i] - x[k])
                y2 = np.polymul(y2, (c / d))
        base_functions1.append(y2)
        base_functions.append(y2)
        y2 = np.poly1d([1])
    for j in range(x.size):
        li = np.polymul(y[j], base_functions1.pop(0))
        polynomial += li
    # print(base_functions)
    # print(polynomial.coefficients)
    # TODO: Generate Lagrange base polynomials and interpolation polynomial

    return polynomial, base_functions


def hermite_cubic_interpolation(x: np.ndarray, y: np.ndarray, yp: np.ndarray) -> list:
    """
    Compute hermite cubic interpolation spline

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points
    yp: derivative values of interpolation points

    Returns:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size == yp.size)
    spline = []
    c = 0

    for i in range(x.size - 1):
        l = np.zeros(4)
        c = i
        a = np.zeros((4, 4))
        for j in range(2):
            a[j] = x[c] ** 3, x[c] ** 2, x[c] ** 1, x[c] ** 0
            l[j] = y[c]
            c = c + 1
        c = i
        for ij in range(2, 4):
            a[ij] = 3 * x[c] ** 2, 2 * x[c] ** 1, x[c] ** 0, 0
            l[ij] = yp[c]
            c = c + 1
        # TODO compute piecewise interpolating cubic polynomials
        spline.append(np.poly1d(np.linalg.solve(a, l)))
    # print(spline[0])
    return spline


####################################################################################################
# Exercise 2: Animation

def natural_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Intepolate the given function using a spline with natural boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO construct linear system with natural boundary conditions
    lan = np.zeros(((x.size - 1) * 4))
    for i in range(x.size - 1):
        lan[i * 4] = y[i]
        lan[i * 4 + 1] = y[i + 1]

    tkn = 0
    t = (x.size - 1) * 4
    c = 0
    d = 0
    a = np.zeros((t, t))
    for i in range(x.size - 1):
        d = 3
        for j in range(4):
            c = i * 4
            a[c][c + j] = x[i] ** d
            d = d - 1
    for i in range(x.size - 1):
        fouri = i * 4
        d = 3
        k = -1
        for j in range(4):
            c = fouri + 1
            a[c][c + k] = x[i + 1] ** d
            k = k + 1
            d = d - 1
        if i < x.size - 2:
            mk = 0
            km = 3
            kk = 2
            for ir in range(3):
                a[fouri + 2][fouri + mk] = km * (x[i + 1] ** kk)
                mk = mk + 1
                km = km - 1
                kk = kk - 1
            mk = 4
            km = -3
            kk = 2
            for am in range(3):
                a[fouri + 2][fouri + mk] = km * (x[i + 1] ** kk)
                km = km + 1
                mk = mk + 1
                kk = kk - 1
            a[fouri + 3][fouri] = 6 * x[i + 1]
            a[fouri + 3][fouri + 1] = 2

            a[4 * i + 3][4 * i + 4] = -6 * x[i + 1]
            a[4 * i + 3][4 * i + 5] = -2

    a[-2][0] = 6 * x[0]
    a[-2][1] = 2

    a[-1][-4] = 6 * x[x.size - 1]
    a[-1][-3] = 2
    # TODO solve linear system for the coefficients of the spline
    # print(a)
    # print(b.shape)
    spline = []
    klm = np.linalg.solve(a, lan)
    # TODO extract local interpolation coefficients from solution
    for i in range(x.size - 1):
        la = i * 4
        spline.append(np.poly1d(klm[la:la + 4]))

    return spline


def periodic_cubic_interpolation(x: np.ndarray, y: np.ndarray) -> list:
    """
    Interpolate the given function with a cubic spline and periodic boundary conditions.

    Arguments:
    x: x-values of interpolation points
    y: y-values of interpolation points

    Return:
    spline: list of np.poly1d objects, each interpolating the function between two adjacent points
    """

    assert (x.size == y.size)
    # TODO: construct linear system with periodic boundary conditions

    # TODO solve linear system for the coefficients of the spline
    lan = np.zeros(((x.size - 1) * 4))
    for i in range(x.size - 1):
        lan[i * 4] = y[i]
        lan[i * 4 + 1] = y[i + 1]

    tkn = 0
    t = (x.size - 1) * 4
    c = 0
    d = 0
    a = np.zeros((t, t))
    for i in range(x.size - 1):
        d = 3
        for j in range(4):
            c = i * 4
            a[c][c + j] = x[i] ** d
            d = d - 1
    for i in range(x.size - 1):
        fouri = i * 4
        d = 3
        k = -1
        for j in range(4):
            c = fouri + 1
            a[c][c + k] = x[i + 1] ** d
            k = k + 1
            d = d - 1
        if i < x.size - 2:
            mk = 0
            km = 3
            kk = 2
            for ir in range(3):
                a[fouri + 2][fouri + mk] = km * (x[i + 1] ** kk)
                mk = mk + 1
                km = km - 1
                kk = kk - 1
            mk = 4
            km = -3
            kk = 2
            for am in range(3):
                a[fouri + 2][fouri + mk] = km * (x[i + 1] ** kk)
                km = km + 1
                mk = mk + 1
                kk = kk - 1
            a[fouri + 3][fouri] = 6 * x[i + 1]
            a[fouri + 3][fouri + 1] = 2

            a[fouri + 3][fouri + 4] = -6 * x[i + 1]
            a[fouri + 3][fouri + 5] = -2
    abc = 3
    de = 2
    for i in range(3):
        a[-2][i] = abc * (x[0] ** de)
        abc = abc - 1
        de = de - 1

    a[-1][0] = 6 * x[0]
    a[-1][1] = 2
    brk = -3
    eks = -4
    two = 2
    for j in range(3):
        a[-2][eks] = brk * (x[x.size - 1] ** two)
        brk = brk + 1
        eks = eks + 1
        two = two - 1

    a[-1][-4] = -6 * x[x.size - 1]
    a[-1][ -3] = -2


    spline = []
    # TODO extract local interpolation coefficients from solution
    klm = np.linalg.solve(a, lan)
    for i in range(x.size - 1):
        la = i * 4
        spline.append(np.poly1d(klm[la:la + 4]))
    return spline


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    a = np.array([[1], [2], [3], [8]])
    b = np.array([[1], [2], [2], [5]])
    c = np.array([[1], [2], [2], [6]])
    # poly1d,base_func=lagrange_interpolation(a,b)
    # spl = hermite_cubic_interpolation(a, b, c)
    #spl = natural_cubic_interpolation(a, b)
    akl=periodic_cubic_interpolation(a,b)
