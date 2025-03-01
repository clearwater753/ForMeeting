# 合并升序数组
class Solution:
    def mergeSortedArray(self, A, B):
        # write your code here
        res = []
        i, j = 0, 0
        while i < len(A) and j < len(B):
            if A[i] < B[j]:
                res.append(A[i])
                i += 1
            else:
                res.append(B[j])
                j += 1
        res.extend(A[i:])
        res.extend(B[j:])  # 两个数组可能长度不一样
        return res  
    