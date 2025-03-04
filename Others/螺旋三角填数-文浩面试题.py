# 只考虑奇数的情况
Dirs = [(1, -1), (0, 1), (-1, -1), (0, -1)]
def f(n):
    m = (n+1) // 2 # 行数
    res = [[0]*n for i in range(m)]
    # 初始点为(0, n//2)
    res[0][n//2] = 1
    # 生成列表
    ord = [0]*m
    for i in range(m):
        ord[i] = 2*i + 1
    nums = list(range(1, sum(ord)+1))
    # print(nums)
    flag = [[0]*n for i in range(m)]
    flag[0][n//2] = 1
    i = 0
    j = n//2
    d = 0
    # 生成矩阵
    for num in nums[1:]:
        x = i + Dirs[d][0]
        y = j + Dirs[d][1]
        if x < 0 or x >= m or y < 0 or y >= n or flag[x][y]==1:
            d = (d+1) % 4
            i = i + Dirs[d][0]
            j = j + Dirs[d][1]
            res[i][j] = num
            flag[i][j] = 1
        else:
            i = x
            j = y
            res[i][j] = num
            flag[i][j] = 1
    return res

result = f(11)
for row in result:
    print(row)

        
