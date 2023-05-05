import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination


def compare_transpose(a: np.ndarray) -> (bool):
    (n, m) = a.shape
    for i in range(n):
        for j in range(n):
            b = a.transpose()
            boo = almost_equal(a[i][j], b[i][j], n)
            if not boo:
                return False
    return True


def back_sub(L: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    y = np.zeros(n)
    print(b)

    y[0] = b[0] / L[0][0]
    for i in range(1, n):
        xi = 0
        for j in range(1, i+1):
            xi = xi + y[i - j] * L[i][i-j]
        y[i] = (b[i] - xi) / L[i][i]
    print(y)

    return y


def almost_equal(a: float, b: float, n: int) -> (bool):
    if b > a:
        a = b
        b = a
    if abs(a - b) > n * n * max(abs(a), abs(b) * np.finfo(float).eps):
        return False
    else:
        return True


def swap_rows(a: np.ndarray, b: int, c: int):
    k = a.copy()
    temp1 = k[b]
    temp2 = k[c]
    a[b] = temp2
    a[c] = temp1


def use_pivot(A: np.ndarray, b: np.ndarray, n: int):
    a = np.zeros(n)
    k = 0
    t = 0
    smallest = A[0][0]
    largest = 0
    for l in range(n):
        a[l] = A[l][0]
        if A[l][0] < smallest:
            smallest = A[l][0]
    for i in range(n):
        if a[i] == smallest:
            t = i
            for j in range(i, n):
                if a[j] > largest and j > i:
                    largest = a[j]
                    k = j
            swap_rows(A, k, i)
            swap_rows(b, k, i)


def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified

    A = A.copy()
    b = b.copy()

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    (n, m_a) = np.shape(A)
    m_b = np.size(b)

    l = 1

    # Test if shape of matrix and vector is compatible
    if m_a != m_b or n != m_a:
        raise ValueError
    if use_pivoting == False:
        for pv in range(m_b - 1):
            if A[pv][0] == 0:
                raise ValueError
    kn = 0
    kl = 0
    # pivoting beginning
    if use_pivoting == True:
        # while A[m_a-1][0]!=0:
        use_pivot(A, b, n)
        # kl=kl+1
        # if kl==m_a-1:
        # break
        kn = 1
    # pivoting end
    for k in range(m_b - kn):
        a_1 = A[k][k]
        b_1 = b[k]
        for i in range(k + 1, m_a):
            for j in range(k + 1, m_a):
                A[i][j] = A[i][j] - A[i][k] / a_1 * A[k][j]
            b[i] = b[i] - A[i][k] / a_1 * b_1
        for t in range(1, m_a):
            A[t][0] = 0
    for di in range(1, m_b):
        for dj in range(0, di):
            A[di][dj] = 0
    for un in range(m_b):
        say = 0
        for uni in range(m_b):
            if A[un][uni] == 0:
                say = say + 1

    # TODO: Perform gaussian elimination

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # TODO: Test if shape of matrix and vector is compatible and raise ValueError if not
    A = A.copy()

    b = b.copy()
    (n, m) = np.shape(A)
    k = np.size(b)
    if m != k or n != m:
        raise ValueError
    if A[n - 1, n - 1] == 0:
        raise ValueError
    # TODO: Initialize solution vector with proper size
    x = np.zeros(n)

    # TODO: Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    x[n - 1] = b[n - 1] / A[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        xi = 0
        for j in range(i, n - 1):
            xi = xi + A[i][j + 1] * x[j + 1]
        x[i] = (b[i] - xi) / A[i][i]
    return x


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # TODO check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if n != m:
        raise ValueError()
    tr = np.transpose(M)
    for i in range(n):
        for j in range(n):
            if not np.isclose(tr[i][j], M[i][j], n):
                raise ValueError

   

    # TODO build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            lk = 0
            lij = 0
            ant = 0
            if i == j:
                for k in range(i):
                    lk = lk + L[i][k] ** 2
                ant = M[i][i] - lk
                if ant < 0:
                    raise ValueError
                L[i][i] = np.sqrt(ant)
            if i > j:
                for k in range(j):
                    lij = lij + L[i][k] * L[j][k]
                L[i][j] = (M[i][j] - lij) / L[j][j]
    akl = np.dot(L, np.transpose(L))

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """
    L = L.copy()

    b = b.copy()
    # TODO Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    z = b.size
    if n != m or z != n:
        raise ValueError
    for i in range(n):
        for j in range(n):
            if i < j and L[i][j] != 0:
                raise ValueError
    # TODO Solve the system by forward- and backsubstitution

    L1 = np.transpose(L)


    y = back_sub(L, b, n)
    x = back_substitution(L1, y)

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    L : system matrix
    g : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    L = np.zeros((n_shots*n_rays, n_grid*n_grid))
    # TODO: Initialize intensity vector
    #kk = np.zeros((n_rays,n_shots))
    g=np.zeros(n_shots * n_rays )

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    for i in range(n_shots):
        theta = np.pi*(i/n_shots)
        intensities, ray_indices, isect_indices, lengths = tomograph.take_measurement(n_grid, n_rays, theta)
        #print(intensities, ray_indices, isect_indices, lengths )
        for k in range(np.size(lengths)):
            L[i*n_rays+ray_indices[k],isect_indices[k]]=lengths[k]
        for j in range(n_rays):
            g[i * n_rays + j]= intensities[j]

    # Take a measurement with the tomograph from direction r_theta.
    # intensities: measured intensities for all <n_rays> rays of the measurement. intensities[n] contains the intensity for the n-th ray
    # ray_indices: indices of rays that intersect a cell
    # isect_indices: indices of intersected cells
    # lengths: lengths of segments in intersected cells
    # The tuple (ray_indices[n], isect_indices[n], lengths[n]) stores which ray has intersected which cell with which length. n runs from 0 to the amount of ray/cell intersections (-1) of this measurement.

    return [L, g]


def compute_tomograph(n_shots: np.int, n_rays: np.int, n_grid: np.int) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of different shot directions
    n_rays   : number of parallel rays per direction
    n_grid   : number of cells of grid in each direction, in total n_grid*n_grid cells

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, g] = setup_system_tomograph(n_shots, n_rays, n_grid)


    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)


    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
    a = np.array([[1,2,3], [2,5,9], [3,9,0]])
    b = np.array([[4], [12], [18],[5]])
    #c = np.array([[8, 1, -9], [9, -3, 3], [-5, 7, -8]])
    #d = np.array([[5], [9], [8]])

    #print(gaussian_elimination(a,b,False))
    # k=back_substitution(c,d)
    # print(k)
    # print(compare_transpose(a))
    # swap_rows(a,0,1)
    # print(a)
    #(f,h)=gaussian_elimination(a,b,False)
    #print(back_substitution(f,h))
    a = (compute_cholesky(a))
    #print(solve_cholesky(e, b))
    #aa=np.int(2)
    #bb=np.int(4)
    #cc=np.int(2)
    #compute_tomograph(aa,bb,cc)
    #setup_system_tomograph(aa,bb,cc)