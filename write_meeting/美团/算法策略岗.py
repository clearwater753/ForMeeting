import numpy as np

nums = [[4, 0, 1, 5, 1, 0, 0],
[5, 5, 4, 0, 2, 0, 1],
[0, 2, 0, 2, 4, 5, 0],
[3, 3, 4, 0, 3, 0, 2],
[4, 0, 1, 4, 0, 0, 5]]
nums = np.array(nums)
# 先求分母
sum_cor = np.array([0]*nums.shape[0])
matrix = np.array([[0]*nums.shape[0]]*nums.shape[0])
for i in range(nums.shape[0]):
    tmp = 0
    for j in range(i, nums.shape[0]):
        correlation = np.corrcoef(nums[i,:], nums[j,:])[0, 1]
        matrix[i, j] = correlation
        matrix[j, i] = correlation
        tmp += correlation
    sum_cor[i] = tmp

# 再求分子
res = np.array([[0]*nums.shape[0]]*nums.shape[0])
for i in range(nums.shape[0]):
    for j in range(nums.shape[1]):
        if nums[i,j] == 0:
            res[i,j] = np.sum(nums[:,j]*matrix[i,:])/sum_cor[i]
print(res)

# n = int(input())
# nums = [int(i) for i in input().split()]
# # nums.sort()
# # 首先确定可以选择的完全平方数的范围
# m = 1
# sort_nums = sorted(nums)
# max_ = sort_nums[-1]*sort_nums[-2]*sort_nums[-3]
# while True:
#     if m*m > max_:
#         break
#     m += 1
# m -= 1
# # 然后遍历所有可能的组合
# for x in range(1, m+1):
#     for i in range(0, n-2):
#         for j in range(i+1, n-1):
#             for k in range(j+1, n):
#                 if x*x == nums[i]*nums[j]*nums[k]:
#                     print(i+1, j+1, k+1)
#                     exit()
# print(-1)
























# T = int(input())
# t = []
# p = 0
# for i in range(T):
#     s = input()
#     for x in s:
#         if '0'<=x<='9':
#             if p == 0:
#                 p = int(x)
#             else:
#                 p = 10*p + int(x)
#         elif 'a'<=x<='z' or 'A'<=x<='Z':
#             # 当s的第i个字符不为数字时，现将字符串t的p个字符进行旋转，然后根据x的值进行操作，如果p越界呢
#             if p != 0:
#                 tmp = p % len(t)
#                 t[:] = t[tmp:] + t[:tmp]
#             if x == 'R':
#                 t = t[::-1]
#             else:
#                 t.append(x)
#     print("".join(t))
# # D0ame3