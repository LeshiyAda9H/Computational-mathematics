from tyt import *

def SQF(a, b, quadFormula, presentValue, l, k_opt = 1, m_original = 3, err_min = 1e-10, err_max = 1e6):
    k = k_opt

    S = np.array([])
    H = np.array([])

    count = 0

    while True:
        count += 1

        s, h = findVal_integral(a, b, k, quadFormula)

        S = np.append(S, s)
        H = np.append(H, h)

        r = len(S) - 1

        if r < 2:
            m = m_original
        else:
            m = order_error_aitken(S, r, l)

        C = richardson(S, H, m, r)

        if r == 0:
            err = abs(presentValue - S[r])
        else:
            err = abs(C[r] - S[r])

        print()
        print("─────────── ⋆⋅♚⋅⋆ ──", "Итерация =", count, "─── ⋆⋅♛⋅⋆ ─────────────")
        print("☞Значение шага:", k)
        print("☞Значение квадратурной формулы:", S[r])
        print("☞Ошибка по Ричардсону:", err)
        print("☞Порядок главного члена погрешности:", m)
        print("☞Действительное значение ошибки:", abs(presentValue - S[r]))
        print("─────────────────────────────────────────────────────────")
        print()

        if err < err_min or err > err_max:
            break

        k *= l

    return S, H
