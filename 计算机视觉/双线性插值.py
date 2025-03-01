import numpy as np

def bilinear_interpolation(x, y, points):
    '''双线性插值
    x, y 是要插值的点的坐标
    points 是一个包含四个点的列表，每个点是 (x, y, value) 的形式
    '''
    points = sorted(points)  # 按 x 排序
    (x1, y1, q11), (x1, y2, q12), (x2, y1, q21), (x2, y2, q22) = points

    if x1 == x2 or y1 == y2:
        raise ValueError("输入点不能形成矩形")

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1))

if __name__ == "__main__":
    # 示例点 (x, y, value)
    points = [(1, 1, 1), (1, 2, 2), (2, 1, 3), (2, 2, 4)]
    x, y = 1.5, 1.5
    value = bilinear_interpolation(x, y, points)
    print(f"插值结果: {value}")