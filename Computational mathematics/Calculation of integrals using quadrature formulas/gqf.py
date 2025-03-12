from tyt import *

def GQF(a, b):
    mu0 = mu_0(b) - mu_0(a)
    mu1 = mu_1(b) - mu_1(a)
    mu2 = mu_2(b) - mu_2(a)
    mu3 = mu_3(b) - mu_3(a)
    mu4 = mu_4(b) - mu_4(a)
    mu5 = mu_5(b) - mu_5(a)

    MU = np.array([[mu0, mu1, mu2], [mu1, mu2, mu3], [mu2, mu3, mu4]])

    mu = np.array([-mu3, -mu4, -mu5])

    A = np.linalg.solve(MU, mu)

    x = np.roots(np.flip(np.append(A, 1)))

    X = np.array([x ** i for i in range(3)])

    mu = np.array([mu0, mu1, mu2])

    A = np.linalg.solve(X, mu)

    return sum([A[i] * f(x[i]) for i in range(3)])
