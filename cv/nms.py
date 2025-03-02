# 非极大值抑制（NMS）流程：
# 1 首先我们需要设置两个值：一个Score的阈值，一个IOU的阈值。
# 2 对于每类对象，遍历该类的所有候选框，过滤掉Score值低于Score阈值的候选框，并根据候选框的类别分类概率进行排序：A < B < C < D < E < F。
# 3 先标记最大概率矩形框F是我们要保留下来的候选框。
# 4 从最大概率矩形框F开始，分别判断A～E与F的交并比（IOU）是否大于IOU的阈值，假设B、D与F的重叠度超过IOU阈值，那么就去除B、D。
# 5 从剩下的矩形框A、C、E中，选择概率最大的E，标记为要保留下来的候选框，然后判断E与A、C的重叠度，去除重叠度超过设定阈值的矩形框。
# 6 就这样重复下去，直到剩下的矩形框没有了，并标记所有要保留下来的矩形框。
# 7 每一类处理完毕后，返回步骤二重新处理下一类对象。

import numpy as np

def nms(boxes, scores, thresh):
  # boxes：(N, 4)
  # scores：(N,)
  x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
  areas = (x2-x1) * (y2-y1)
  order = scores.argsort()[::-1]
  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    x1_inter = np.maximum(x1[i], x1[order[1:]]) #广播机制
    y1_inter = np.maximum(y1[i], y1[order[1:]])
    x2_inter = np.minimum(x2[i], x2[order[1:]])
    y2_inter = np.minimum(y2[i], y2[order[1:]])
    
    inter = np.maximum(0, x2_inter-x1_inter)*np.maximum(0, y2_inter-y1_inter)
    ious = inter / (areas[i] + areas[order[1:]] - inter)
    idxs = np.where(ious < thresh)[0]
    order = order[1:][idxs]
  return keep

# 示例使用
if __name__ == "__main__":
    boxes = np.array([
        [100, 100, 210, 210],
        [105, 105, 215, 215],
        [300, 300, 400, 400]
    ])

    scores = np.array([0.9, 0.85, 0.95])
    iou_threshold = 0.5

    keep_indices = nms(boxes, scores, iou_threshold)
    print("保留的边界框索引:", keep_indices)
    print("保留的边界框:", boxes[keep_indices])