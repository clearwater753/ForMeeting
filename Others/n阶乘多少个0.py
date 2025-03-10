def f(n):
    assert n > 0 and isinstance(n, int)
    num = 1
    for i in range(1, n+1):
        num *= i
    num = str(num)
    res = 0
    for i in range(len(num)-1, -1, -1):
        if num[i] == '0':
            res += 1
        else:
            break
    return res

print(f(100))

def f2(n):
    res = 0
    for i in range(1, n+1):
        while i % 5 == 0:
            res += 1
            i //= 5
    return res

print(f2(100))
    


