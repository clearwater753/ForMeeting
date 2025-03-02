
def move(n, a,b,c): #n代表a柱的盘子数
    if n == 1: #Base case,递归的结束条件
        print(f"{a}->{c}")
        return
    
    move(n-1,a, c, b) # 将A柱上面（n-1）个盘子作为整体从A柱移动到B柱
    move(1, a, b, c) # 将A柱的最后一个盘子从A柱移动到C柱
    move(n-1, b, a, c) # 将B柱的（n-1）个盘子从B柱移动到C柱

move(4,'a','b','c')