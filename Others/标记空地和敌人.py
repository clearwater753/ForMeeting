# 给你一个二维矩阵，其中空地用'.'表示， 敌人用'@'表示，
# 对于每个d*d的方块(1<=d<=min(m,n)), 
# 如果这个方块都是空地，那么这个方块就是安全的，请输出安全的方块的个数。
m, n = 3, 3
matrix = [
    ['.', '.', '.'],
    ['.', '.', '.'],
    ['.', '.', '.']
]
flag = [[0]*n for _ in range(m)]
cnt = 0

def check(i, j, d=1):
    if i + d >= m or j + d >= n:
        return d
    for x in range(i, i+d+1):
        if matrix[x][j+d] == '@':
            return d
    for y in range(j, j+d+1):
        if matrix[i+d][y] == '@':
            return d
    d += 1
    res = check(i, j, d)
    return res

for i in range(m):
    for j in range(n):
        if matrix[i][j] == '.' and flag[i][j] == 0:
            # 检查这个方块是否都是空地
            d = check(i, j)
            for x in range(i, i+d):
                for y in range(j, j+d):
                    flag[x][y] = 1
            cnt += 1
        else:
            continue
print(cnt)


