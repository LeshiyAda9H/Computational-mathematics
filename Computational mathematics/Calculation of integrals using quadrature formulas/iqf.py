from tyt import *

def IQF(a, b):
    x1 = a
    x2 = (a + b) / 2
    x3 = b
    x = np.array([x1, x2, x3])

    mu0 = mu_0(b) - mu_0(a)
    mu1 = mu_1(b) - mu_1(a)
    mu2 = mu_2(b) - mu_2(a)
    mu = np.array([mu0, mu1, mu2])

    A = x ** np.arange(3)[:, None]
    A_values = np.linalg.solve(A, mu)

    return sum([A_values[i] * f(x[i]) for i in range(3)])



