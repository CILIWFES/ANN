from main.dataprocessing.training_set import *
from main.dataprocessing.image import IMP
import numpy as np
#
# # 读入数据
# (train_img, train_lbl, train_rows, train_cols) = FsMnist.read_TrainImg()
# (val_img, val_lbl, test_rows, test_cols) = FsMnist.read_TestImg()
# train_img = train_img[0:15]
# # 生成增强的图像，最佳方法是在另一进程执行，这里只是演示
# # 首先复制一份原始图像
# aug_img = train_img.copy()
# # 修改其中的每幅图像
# for i in range(aug_img.shape[0]):
#     # 有50%概率做左右翻转
#     if np.random.random() < 0.5:
#         # aug_img[i][0]为第i号样本的0号通道，灰度图像只有0号通道
#         # fliplr()用于左右翻转
#         aug_img[i][0] = np.fliplr(aug_img[i][0])
#
#     # 左右移动最多2个像素，注意randint(a,b)的范围为a到b-1
#     amt = np.random.randint(0, 3)
#     if amt > 0:  # 如果需要移动…
#         if np.random.random() < 0.5:  # 左移动还是右移动？
#             # pad()用于加上外衬，因移动后减少的区域需补零
#             # 然后用[:]取所要的部分
#             aug_img[i][0] = np.pad(aug_img[i][0], ((0, 0), (amt, 0)), mode='constant')[:, :-amt]
#         else:
#             aug_img[i][0] = np.pad(aug_img[i][0], ((0, 0), (0, amt)), mode='constant')[:, amt:]
#
#     # 上下移动最多2个像素
#     amt = np.random.randint(0, 3)
#     if amt > 0:
#         if np.random.random() < 0.5:
#             aug_img[i][0] = np.pad(aug_img[i][0], ((amt, 0), (0, 0)), mode='constant')[:-amt, :]
#         else:
#             aug_img[i][0] = np.pad(aug_img[i][0], ((0, amt), (0, 0)), mode='constant')[amt:, :]
#
#     # 随机清零最大5*5的区域
#     x_size = np.random.randint(1, 6)
#     y_size = np.random.randint(1, 6)
#     x_begin = np.random.randint(0, 28 - x_size + 1)
#     y_begin = np.random.randint(0, 28 - y_size + 1)
#     aug_img[i][0][x_begin:x_begin + x_size, y_begin:y_begin + y_size] = 0
#
# imp = IMP()
# imp.whirl(aug_img[0],0,0,0)
