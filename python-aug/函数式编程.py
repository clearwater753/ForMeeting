# from functools import reduce

# def str2float(s):
#     def fn(x, y):
#         return x * 10 + y
#     def char2num(s):
#         return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]
#     n = s.index('.')
#     s1 = list(map(char2num, s[:n]))
#     s2 = list(map(char2num, s[n+1:]))
#     return reduce(fn, s1) + reduce(fn, s2) / 10**len(s2)

# print('str2float(\'123.456\') =', str2float('123.456'))
# if abs(str2float('123.456') - 123.456) < 0.00001:
#     print('测试成功!')
# else:
#     print('测试失败!')
# print(str.lower('Hello'))
# def is_palindrome(n):
#     return str(n) == str(n)[::-1]

# # 测试:
# output = filter(is_palindrome, range(1, 1000))
# print('1~1000:', list(output))
# if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]:
#     print('测试成功!')
# else:
#     print('测试失败!')

L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

def by_name(t):
    return t[1]

L2 = sorted(L, key=by_name, reverse=True)
print(L2)