import numpy as np

a = 2.1
b = 3.3

metLogical_err = 3.9944014305257426568
exactVal_integral_comparison = 4.461512705331194112840828080521604042844


def findVal_integral(a, b, k, quad_formula):
    s = 0
    h = (b - a) / k
    for i in range(k):
        z_i = a + h * i
        s += quad_formula(z_i, z_i + h)

    return s, h

def order_error_aitken(S, r, l):
    return -np.log(abs(S[r] - S[r-1]) / abs(S[r-1] - S[r-2])) / np.log(l)

def richardson(S, H, m, r):
    A = np.empty((r + 1, r + 1))
    for i in range(r):
        A[::, i] = pow(H, i + m)
    A[::, r] = -1
    C = np.linalg.solve(A, -S)
    return C

def k_opt(S, H, l, err, a, b):
    m = order_error_aitken(S, 2, l)
    Rh1 = (S[1] - S[0]) / (1 - l ** (-m))
    h_opt = H[0] * ((err / abs(Rh1)) ** (1 / m))

    return int((b - a) / (h_opt * 0.95))

def p(x):
    return (x - 2.1) ** (- 2 / 5)
def f(x):
    return 4.5 * np.cos(7 * x) * np.exp(-2 * x / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3
def mu_0(x):
    return (5 / 3) * (x - a) ** (3 / 5)
def mu_1(x):
    return (5 / 24) * ((x - a) ** (3 / 5)) * (5 * a + 3 * x)
def mu_2(x):
    return (5 / 156) * ((x - a) ** (3 / 5)) * (25 * (a ** 2) + 15 * a * x + 12 * (x ** 2))

def mu_3(x):
    return (5 * ((x - a) ** (3 / 5)) * (125 * (a ** 3) + 75 * (a ** 2) * x + 60 * a * (x ** 2) + 52 * (x ** 3))) / 936

def mu_4(x):
    return (5 * ((x - a) ** (3 / 5)) * (625 * (a ** 4) + 375 * (a ** 3) * x + 300 * (a ** 2) * (x ** 2) + 260 * a * (x ** 3) + 234 * (x ** 4))) / 5382

def mu_5(x):
    return (5 * ((x - a) ** (3 / 5)) * (15625 * (a ** 5) + 9375 * (a ** 4) * x + 7500 * (a ** 3) * (x ** 2) + 6500 * (a ** 2) * (x ** 3) + 5850 * a * (x ** 4) + 5382 * (x ** 5))) / 150696

