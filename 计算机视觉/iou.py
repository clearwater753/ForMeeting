def calculate_iou(box1, box2):
    # [left, top, right, bottom]
    # 计算交集
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算每个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集
    union_area = box1_area + box2_area - inter_area

    # 计算 IoU
    iou = inter_area / union_area

    return iou

# 示例
box1 = [1, 1, 3, 3]
box2 = [2, 2, 4, 4]
iou = calculate_iou(box1, box2)
print(f"IoU: {iou:.4f}")