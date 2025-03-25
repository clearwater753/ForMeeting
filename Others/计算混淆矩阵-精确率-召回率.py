import numpy as np

def calculate_confusion_matrix(y_true, y_pred, num_lables):
    # 初始化混淆矩阵
    cm = np.zeros((num_lables, num_lables), dtype=int)
    # 填充混淆矩阵
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def calculate_metrics(cm):
    # 计算各类别的 TP、FP、TN 和 FN
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    FP = np.sum(cm, axis=0) - TP
    TN = np.sum(cm) - TP - FP - FN # 广播机制
    precise = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precise, recall

# 示例用法
y_true = np.random.randint(0, 10, 1000)
y_pred = np.random.randint(0, 10, 1000)
num_classes = 10
cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
precise, recall = calculate_metrics(cm)
print(f'{precise}', '\n', f'{recall}')
