def fast_power(x, n):
    if n == 0:
        return 1
    half_power = fast_power(x, n // 2)
    if n % 2 == 0:
        return half_power * half_power
    else:
        return x * half_power * half_power

# 示例用法
x = 2
n = 10
result = fast_power(x, n)
print(f"{x} 的 {n} 次幂是 {result}")
    
