from sqf import *
from iqf import *
from gqf import *

print("☯" * 65)
print("(｡◕ ‿ ◕｡)" * 5, "IQF", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 65)

IQF_val = IQF(a, b)

print()
print("IQF: ", IQF_val)
print("Методическая погрешность: ", metLogical_err)
print("Реальная погрешность: ", abs(exactVal_integral_comparison - IQF_val))
print()

print("☯" * 68)
print("(｡◕ ‿ ◕｡)" * 5, "SQF_IQF", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 68)

S, H = SQF(a, b, IQF, exactVal_integral_comparison, 2)
SQF_val = S[-1]

print("SQF_IQF: ", SQF_val)
print("Реальная погрешность: ", abs(exactVal_integral_comparison - SQF_val))
print()

print("☯" * 76)
print("(｡◕ ‿ ◕｡)" * 5, "SQF_IQF с k_opt", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 76)

k0 = k_opt(S, H, 2, 1e-6, a, b)
S, H = SQF(a, b, IQF, exactVal_integral_comparison, 2, k0)
SQF_val_k_opt = S[-1]

print("SQF_IQF c k_opt: ", SQF_val_k_opt)
print("Реальная погрешность с оптимальным шагом: ", abs(exactVal_integral_comparison - SQF_val_k_opt))
print()

print("☯" * 65)
print("(｡◕ ‿ ◕｡)" * 5, "GQF", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 65)

GQF_val = GQF(a, b)

print()
print("GQF: ", GQF_val)
print("Реальная погрешность: ", abs(exactVal_integral_comparison - IQF_val))
print()

print("☯" * 68)
print("(｡◕ ‿ ◕｡)" * 5, "SQF_GQF", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 68)

S, H = SQF(a, b, GQF, exactVal_integral_comparison, 2, 1, 6)
SQF_val = S[-1]

print("SQF_GQF: ", SQF_val)
print("Реальная погрешность: ", abs(exactVal_integral_comparison - SQF_val))
print()

print("☯" * 76)
print("(｡◕ ‿ ◕｡)" * 5, "SQF_GQF c k_opt", "(｡◕ ‿ ◕｡)" * 5)
print("☯" * 76)

k0 = k_opt(S, H, 2, 1e-6, a, b)
S, H = SQF(a, b, GQF, exactVal_integral_comparison, 2, k0, 6)
SQF_val_k_opt = S[-1]

print("SQF_GQF c k_opt: ", SQF_val_k_opt)
print("Реальная погрешность с оптимальным шагом: ", abs(exactVal_integral_comparison - SQF_val_k_opt))
print()