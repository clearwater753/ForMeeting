# def trim(s):
#     left = 0
#     right = len(s) - 1
#     while left <= right:
#         if s[left] == ' ':
#             left += 1
#         elif s[right] == ' ':
#             right -= 1
#         else:
#             break
#     return s[left:right+1]
# # 测试:
# if trim('hello  ') != 'hello':
#     print('测试失败!')
# elif trim('  hello') != 'hello':
#     print('测试失败!')
# elif trim('  hello  ') != 'hello':
#     print('测试失败!')
# elif trim('  hello  world  ') != 'hello  world':
#     print('测试失败!')
# elif trim('') != '':
#     print('测试失败!')
# elif trim('    ') != '':
#     print('测试失败!')
# else:
#     print('测试成功!')

# 

L1 = ['Hello', 'World', 18, 'Apple', None]
L2 = [ s.lower() for s in L1 if isinstance(s, str)]

# 测试:
print(L2)
if L2 == ['hello', 'world', 'apple']:
    print('测试通过!')
else:
    print('测试失败!')