n, m = map(int, input().split())
nums = [int(s) for s in input().split()]
if 2 * m < n:
    print(-1)

# dp[i][j]表示第i分钟阅读到第j页的最大知识量,dp[m][n]
# dp[i][j] = max(dp[i-1][j-1] + 1, dp[i-1][j-2]+1)
# dp初始化
dp = [[0]* (n+1) for _ in range(m+1)]
for i in range(1,m+1):
    for j in range(1,n+1):
        if j >= 2:
            dp[i][j] = max(dp[i-1][j-1]+nums[j], dp[i-1][j-2] + (nums[j-1]+nums[j])/2)
        else:
            dp[i][j] = dp[i-1][j-1] + nums[j]

print(dp[m][n])

















# n = int(input())
# nums = [int(s) for s in input().split()]
# res = 0
# cnt = [0]*n
# # 获取前缀和
# # 先考虑不翻转的情况
# prev = 0
# for i in range(n):
#     cnt[i] = prev + nums[i]
#     prev = cnt[i]
#     res = max(res, abs(cnt[i]))
# # 再遍历翻转的情况
# # 可以考虑遍历翻转的位置i求绝对值的最大值, i表示经过i传送后选择翻转
# for i in range(n):
#     cur = -cnt[i]
#     res = max(abs(cur), res)
#     for j in range(i+1, n):
#         cur += nums[j]
#         res = max(abs(cur), res)
# print(res)