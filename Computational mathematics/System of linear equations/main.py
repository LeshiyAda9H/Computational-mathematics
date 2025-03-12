from math import sqrt
import numpy as np

eps = 10e-14


def luMatrix(A):

    n = A.shape[0]
    L = np.identity(n)
    U = A.copy().astype(float)
    P = np.identity(n)
    Q = np.identity(n)

    difP, difQ = 0, 0

    for k in range(n):

        pivot_index = np.argmax(np.abs(U[k:, k:]))

        i = pivot_index // (n - k) + k
        j = pivot_index % (n - k) + k

        if (k != i):
            difP += 1

        if (k != j):
            difQ += 1

        # Меняем местами строки и столбцы в матрицах U и P
        U[[k, i], :] = U[[i, k], :]
        U[:, [k, j]] = U[:, [j, k]]
        P[[k, i], :] = P[[i, k], :]
        Q[:, [k, j]] = Q[:, [j, k]]

        # Вычисляем множители и обновляем матрицы L и U
        U[k + 1:, k] /= U[k, k]
        U[k + 1:, k:] -= np.outer(U[k + 1:, k], U[k, k:])
        # U[k + 1:, k] = L[k + 1:, k]

    # Извлекаем диагональ и верхний треугольник из U
    L = np.tril(U)
    for i in range(n):
        L[i, i] = 1
    U = np.triu(U)
    return P, L, U, Q, difP, difQ

def luColumn(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.copy(A).astype(float)
    P = np.eye(n)
    dif = 0
    for k in range(n):
        # Find the index of the pivot element in column k
        pivot_index = np.argmax(np.abs(U[k:, k])) + k
        if (pivot_index != k):
            dif += 1

        # Swap rows k and pivot_index
        U[[k, pivot_index], :] = U[[pivot_index, k], :]
        P[[k, pivot_index], :] = P[[pivot_index, k], :]

        L[[k, pivot_index], :k] = L[[pivot_index, k], :k]

        # Update the L matrix and U matrix
        if (abs(U[k, k]) < eps * norm(A)):
            continue
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])

    return P, L, U, dif
def det(A):

    n = A.shape[0]
    P, L, U, dif = luColumn(A)

    det_U = np.prod(np.diag(U))
    det_P = (-1) ** dif
    det_L = np.prod(np.diag(L))

    det_A = det_P * det_L * det_U

    return det_A


def luSolve(A, b):

    n = A.shape[0]
    P, L, U, _ = luColumn(A)

    # Ax=b => PAx=Pb => LUx=Pb => Ly=Pb(y=Ux)
    # Solve Ly=b for y

    Pb = np.dot(P, b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        # y[i] = Pb[i] - sum(L[i][j]*y[j] for j in range(i))

    # Solve Ux=y for x
    x = np.zeros((n, 1))

    for i in range(n - 1, -1, -1):
        if (abs(U[i, i]) < eps * norm(A)):
            return None
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        # x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
    return x


def solveAccuracy(A, b):
    dif = A @ luSolve(A, b) - b
    return dif


def luInverse(A):
    n = A.shape[0]
    P, L, U, dif = luColumn(A)

    # Solve AX[:,i] = I[:,i] для каждого столбца i идентификационной матрицы
    # LUX=PAX=PI=> Сначала решаем Ly=PI => решаем UX=y
    X = np.zeros((n, n))

    for i in range(n):

        b = np.zeros(n)
        b[i] = 1.0
        Pb = np.dot(P, b)
        y = np.zeros(n)

        for j in range(n):
            y[j] = Pb[j] - np.dot(L[j, :j], y[:j])
        x = np.zeros(n)

        for j in range(n - 1, -1, -1):
            x[j] = (y[j] - np.dot(U[j, j + 1:], x[j + 1:])) / U[j, j]
        X[:, i] = x

    return X


def inverseAccuracy(A):
    print(A @ luInverse(A), "\t: A*A^(-1)")
    print(luInverse(A) @ A, "\t: A^(-1)*A")


def luCond(A):
    return norm(A) * norm(luInverse(A))


def rank(A):

    U = A.copy().astype(float)
    n = A.shape[0]
    resultRank = n
    i = 0

    while i <= n - 1:

        if (i >= resultRank):
            return resultRank
        j = np.argmax(np.abs(U[i:, i])) + i

        if (abs(U[j][i]) < eps * norm(A)):
            U[:, [i, resultRank - 1]] = U[:, [resultRank - 1, i]]
            resultRank -= 1
            i -= 1
        else:
            U[[i, j], :] = U[[j, i], :]
            U[i + 1:, i:resultRank] -= np.outer((U[i + 1:, i] / U[i, i]), U[i, i:resultRank])
        i += 1

    return resultRank


def luSolve2(A, b):

    # Ax=b=>L*U*inv(Q)x=Pb
    P, L, U, Q, _, _ = luMatrix(A)
    if (abs(det(A)) > eps * norm(A)):
        return luSolve(A, b)

    # Lz=Pb, z =U*inv(Q)x
    n = A.shape[0]
    Pb = np.dot(P, b)
    z = np.zeros((n, 1))

    for i in range(n):
        a = Pb[i]
        v = np.dot(L[i, :i], z[:i])
        z[i] = Pb[i] - np.dot(L[i, :i], z[:i])

    # z=U*y, y = inv(Q)x
    y = np.zeros((n, 1))
    numOfVariables = n

    for i in range(n - 1, -1, -1):

        if (abs(U[i][i]) < eps * norm(U)):

            if (abs(y[i]) > eps * norm(U)):
                return None
            else:
                numOfVariables -= 1
        else:
            # tmp = np.dot(U[i,i+1:(numOfVariables)],x[i+1:(numOfVariables)])
            y[i] = (z[i] - np.dot(U[i, i + 1:], y[i + 1:])) / U[i][i]

    # y = inv(Q)x
    x = Q @ y
    return x


# 2.3 QR


def QR_dec(A):

    N = A.shape[0]
    Q = np.eye(N)
    R = A.copy()

    for iter in range(N):
        y = np.zeros((N, 1))

        for i in range(iter, N):
            y[i][0] = R[i][iter]

        norm_y = np.linalg.norm(y)
        sign = np.sign(R[iter, iter])
        norm_y *= sign

        e = np.zeros((N, 1))
        e[iter][0] = norm_y

        w = y + e
        k = np.linalg.norm(w) ** 2
        mulp = np.dot(w, w.T)
        mulp[iter:, iter:] /= (k * 0.5)

        H = np.eye(N)
        H -= mulp
        R = np.dot(H, R)
        Q = np.dot(Q, H)

    print('Q Q^t = \n', np.dot(Q, np.transpose(Q)))
    print('Q^t Q = \n', np.dot(Q.T, Q))
    print('QR = \n', np.dot(Q, R))
    print('A = \n', A)
    return Q, R


def SLAE(Q, R, b):

    N = Q.shape[0]
    Qtb = Q.T @ b
    x = np.zeros((N, 1))

    for i in range(N - 1, -1, -1):
        x[i, 0] = Qtb[i, 0]

        for j in range(N - 1, i, -1):
            x[i, 0] -= x[j, 0] * R[i, j]
        x[i, 0] /= R[i, i]

    print("Ax =\n", np.dot(A, x))
    print("b = \n", b)


n = 5
# A=np.random.randint(10,size=(n,n))
A = np.array([[-2, 5, 1, 3], [5, 4, 5, 18], [1, 3, 2, 7], [8, -9, 1, 5]])
# A = np.array([[1,3,2],[4,6,5],[7,9,8]])

b = np.random.randint(10, size=(4, 1))
# A=np.random.rand(n,n)
# A[:,2]=A[:,1]+A[:,0]
# A[:,3]=A[:,1]-A[:,0]

# b = np.dot(A, b)

N = 4
MIN = -10
MAX = 10

# define A matrix
A_int = np.random.randint(MIN, MAX + 1, size=(N, N))
A = A_int.astype(np.float64)
# A = np.array([[3., -2., 5.], [7., 4., -8.], [5., -3., -4.]])
# Вычислите QR-декомпозицию A с помощью qr-функции NumPy
Q_np, R_np = np.linalg.qr(A)

# Compute the QR decomposition of A using your QR_dec function
Q, R = QR_dec(A)
print("Q's")
print('Q\n', Q, '\n', 'Q_Numpy\n', Q_np, '\n')
print("R's")
print('R\n', R, '\n', 'R_Numpy\n', R_np, '\n')

# b = np.array([[7.], [3.], [-12.]])
b_int = np.random.randint(MIN, MAX + 1, size=(N, 1))
b = b_int.astype(np.float64)
print("b matrix:\n", b, "\n")

SLAE(Q, R, b)


# 2.4


def equal(A, B, err):
    return np.allclose(A, B, atol=err)


def norm(A):

    res = np.linalg.norm(A, ord=np.inf)

    if res < 0:
        return -res
    else:
        return res


def diag_dom(A):
    return np.all(2 * np.abs(A.diagonal()) >= np.sum(np.abs(A), axis=1))


def seidel(A, b, err):

    N = A.shape[0]
    x = np.zeros((N, 1))
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = -A[i, j] / A[i, i]

    # print("\nC\n",C)
    d = np.zeros((N, 1))
    for i in range(N):
        d[i, 0] = b[i, 0] / A[i, i]

    norm_c = norm(C)

    # норма верхнего треугольника
    err *= (1 - norm_c) / norm_c if norm_c < 1 else 1e-2
    x_tmp = np.zeros((N, 1))

    for iter in range(100000):
        for i in range(N):
            x[i, 0] = 0
            for j in range(i):
                x[i, 0] += C[i, j] * x[j, 0]
            for j in range(i, N):
                x[i, 0] += C[i, j] * x_tmp[j, 0]
            x[i, 0] += d[i, 0]
        # print("\nx\n", x - x_tmp)
        if norm(x - x_tmp) <= err:
            print(norm(np.linalg.solve(A, b) - x))
            return iter + 1
        x_tmp = x.copy()

    return -1


def jacobi(A, b, err):

    N = A.shape[0]
    C = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                C[i, j] = -A[i, j] / A[i, i]

    d = np.zeros((N, 1))

    for i in range(N):
        d[i, 0] = b[i, 0] / A[i, i]

    q = norm(C)
    # print(C)

    # print(np.log(abs(1-q)))
    # print(q,err,norm(d),abs((1 - q)) / err * norm(d),sep=' ')
    apr = np.log((1 - q) * err / norm(d)) / np.log(q)
    # apr = np.log2(err-(1-q)/norm(d))/np.log2(q)
    # apr = (np.log(err)-np.log(norm(d))+np.log(abs(1 - q))) / np.log(q)

    print("\nq: ", q)

    print("Apr iter:", apr)
    if (q > 1):
        err *= 0.01
    else:
        err *= (1 - q) / q
    # print("Matrix C:\n", C)
    # print("Matrix d:\n", d, "\n")
    # print("Norm of C:", q, "\n")

    x_tmp = np.zeros((N, 1))

    for iter in range(10000):

        x = np.dot(C, x_tmp) + d

        if norm(x - x_tmp) <= err:
            print(norm(np.linalg.solve(A, b) - x))
            return iter + 1

        x_tmp = x.copy()

    #  print("\nx\n", x)
    return -1


N = 100
MIN = -10
MAX = 10
err = 1e-10

# define A matrix
A_int = np.random.randint(MIN, MAX + 1, size=(N, N))
# print(A_int)
Diag = np.random.randint(MIN + N * MAX, (N + 1) * MAX, size=(1, N))
# print("Diag\n", Diag)
A_diag = np.diag(Diag[0])
A_int += A_diag

# A_int = (A_int + A_int.T) / 2
# A_int = np.dot(A_int, A_int.T)
A = A_int.astype(np.float64)

# A = np.array([[3., 1., 1.], [1., 5., 1.], [1., 1., 7.]])

# print("A\n", A)

# define b matrix
b_int = np.random.randint(MIN, MAX + 1, size=(N, 1))
b = b_int.astype(np.float64)
# b = np.dot(A, b)
# b = np.array([[5.], [7.], [9.]])

# print("b\n", b)
print()
print('Решение системы по Якоби с диагональной матрицей доминирования за ', jacobi(A, b, err), ' итераций\n')
print('Решение системы по Зайделю с диагональной матрицей доминирования за ', seidel(A, b, err), ' итераций\n')
