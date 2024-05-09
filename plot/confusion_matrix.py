# confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# 标签
classes = ["Approach", "TakeT-tube", "WaitingRobot", "TakePreservCyt", "TwistPreservCyt",
           "TakePipette", "UsePipette", "PlacePipette", "PlacePreservCyt", "Departure"]

# 标签的个数
classNamber = 10  # 分类数量

# 在标签中的矩阵
confusion_matrix = np.array([
    (0.83, 0.08, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04),
    (0.00, 0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    (0.00, 0.00, 0.03, 0.96, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01),
    (0.00, 0.00, 0.03, 0.01, 0.91, 0.00, 0.00, 0.01, 0.00, 0.04),
    (0.00, 0.00, 0.03, 0.01, 0.00, 0.79, 0.00, 0.15, 0.00, 0.02),
    (0.01, 0.00, 0.01, 0.00, 0.00, 0.00, 0.94, 0.02, 0.00, 0.02),
    (0.01, 0.00, 0.04, 0.03, 0.00, 0.01, 0.05, 0.86, 0.00, 0.00),
    (0.00, 0.00, 0.00, 0.26, 0.00, 0.03, 0.00, 0.00, 0.71, 0.00),
    (0.02, 0.00, 0.25, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.69)
], dtype=np.float64)
confusion_matrix = confusion_matrix.T

plt.figure(figsize=(10, 8))

sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=classes, yticklabels=classes,
            annot_kws={'size': 14})

# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
# # plt.title('confusion_matrix')
# plt.colorbar()
# tick_marks = np.arange(len(classes))
# plt.xticks(tick_marks, classes, rotation=-45)
# plt.yticks(tick_marks, classes)
# plt.tight_layout()
#
# thresh = confusion_matrix.max() / 2.
# # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# # ij配对，遍历矩阵迭代器
# iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (confusion_matrix.size, 2))
# for i, j in iters:
#     plt.text(j, i, format(confusion_matrix[i, j]), va='center', ha='center')  # 显示对应的数字
#
plt.xticks(rotation=-40, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.ylabel('True Action', fontsize=30)
plt.xlabel('Predicted Action', fontsize=30)
plt.subplots_adjust(top=0.95, bottom=0.25, left=0.25, right=0.98)
plt.show()
