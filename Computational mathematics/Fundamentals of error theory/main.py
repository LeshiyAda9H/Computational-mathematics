import math
def geron(n, _x):
    return (1 / 2) * (n + _x / n)

def myCos(x, i=0, err=10**9, eps_u = (1 / (2.85 * (10 ** (6)))) ):
    if i < 0:
        return 1
    res = pow(x, 2*i) / math.factorial(2*i)
    if res <= eps_u:
        return (-1)**i * res
    return pow(-1, i) * res + myCos(x, i+1, err=err, eps_u=eps_u)

def func_u(_x):
    number = 2.6 * _x + 0.1
    return myCos(number)

def func_w(_x):
    mySqrt = 0
    tmp = 1
    eps_w = 1 / (10 ** (6))
    result_cos = func_u(_x)

    if(result_cos <= 0):
        return print('bruh')
    else:
        while (abs(geron(tmp, result_cos) - tmp) > eps_w):
            mySqrt = (geron(tmp, result_cos))
            tmp = mySqrt

    return mySqrt

def func_v(_x):
    eps_v = 1 / (0.36 * (10 ** (6)))

    counter = 0
    s = 1
    s_last = 0

    while (abs(s - s_last) > eps_v):
        s_last = s
        s += ((pow(_x, counter + 1)) / (math.factorial(counter + 1)))
        # s_last += ((pow(_x, counter)) / (math.factorial(counter)))

        counter += 1

    return s
def func_z(_x):
    eps_z = 1 / (0.95 * (10 ** (6)))

    return func_w(_x) / func_v(1+_x)
def func_py(_x):
    return (math.sqrt(math.cos(2.6 * _x + 0.1))) / (math.exp(1 + _x))


print("    x   ┃          z(x)          ┃          z`(x)         ┃         |▲z(x)|  ")
print("━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━┃━━━━━━━━━━━━━━━━━━━━━━━━")

x = 0.1

while (x <= 0.2):
    if(x == 0.1):
        print(round(x, 2), "    ┃", func_py(x), "   ┃", func_z(x), "   ┃", abs(func_z(x) - func_py(x)))
    else:
        print(round(x, 2), "   ┃", func_py(x), "   ┃", func_z(x), "   ┃", abs(func_z(x) - func_py(x)))
        ##print(round(x, 2) ,"   ┃", round(func_py(x), 11), "   ┃", round(func_z(x), 11) ,"   ┃", round (abs(func_z(x)-func_py(x)), 11) )

    x += 0.01
