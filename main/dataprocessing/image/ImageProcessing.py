import numpy as np
import math
import matplotlib.pyplot as plt


def rotation_matrix(theta):
    """
    3D 旋转矩阵，围绕X轴旋转theta角
    """
    return np.c_[
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ]


# np.c_[ ] 将列表中的元素在第二维上拼接起来
# np.c_[[1,2],[3,4],[5,6]] =
# array([[1, 3, 5],
#        [2, 4, 6]])
def undo_normalise(im):
    return (1 / (np.exp(-im) + 1) * 257 - 1).astype("uint8")


class ImageProcessing:

    # 1.传入 [图片数, 通道, 高, 宽]
    # 2.传入 [ 通道, 高, 宽]
    def analyse(imgs: np.ndarray):
        i = 0
        size = imgs.shape
        if len(size) == 4:
            i = 1
        elif len(size) == 3:
            i = 0
        else:
            raise Exception("未知参数 尺寸:", size)
        # 通道数,高,宽
        return imgs.shape[i], imgs.shape[i + 1], imgs.shape[i + 2]

    def showGrayscale(self, imgs: np.ndarray, labels: np.ndarray = None):
        # imgs [图片数, 通道, 高, 宽]
        img_size = imgs.shape[0]
        if labels is not None and labels.shape[0] != img_size:
            raise Exception("输入图片与标签不符合")

        channels, high, width = self.analyse(imgs)
        if channels != 1:
            raise Exception("灰度图像通道数不能大于1")

        fig = plt.figure()
        # 向上转型
        matrix_size = math.ceil(pow(img_size, 0.5))
        for i in range(img_size):
            # 添加图层
            ax = fig.add_subplot(matrix_size, matrix_size, i + 1)
            # 设置标签
            if labels is not None:
                ax.set_title(labels[i])
            # 图像展示
            plt.imshow(imgs[i][0], cmap="Greys_r")
            # 坐标轴关闭
            plt.axis("off")
        # 图层展示
        plt.show()

    def changeImage(self, imgs: np.ndarray):
        channels, high, width = self.analyse(imgs)

        for i in range(imgs.shape[0]):
            # 有50%概率做左右翻转
            if np.random.random() < 0.5:
                # fliplr()用于左右翻转
                for j in range(channels):
                    imgs[i][j] = np.fliplr(imgs[i][j])
            # 左右移动最多2个像素，注意randint(a,b)的范围为a到b-1
            amt = np.random.randint(0, 3)
            if amt > 0:  # 如果需要移动…
                if np.random.random() < 0.5:  # 左移动还是右移动？
                    # pad()用于加上外衬，因移动后减少的区域需补零
                    # 然后用[:]取所要的部分
                    imgs[i][0] = np.pad(imgs[i][0], ((0, 0), (amt, 0)), mode='constant')[:, :-amt]
                else:
                    imgs[i][0] = np.pad(imgs[i][0], ((0, 0), (0, amt)), mode='constant')[:, amt:]

            # 上下移动最多2个像素
            amt = np.random.randint(0, 3)
            if amt > 0:
                if np.random.random() < 0.5:
                    imgs[i][0] = np.pad(imgs[i][0], ((amt, 0), (0, 0)), mode='constant')[:-amt, :]
                else:
                    imgs[i][0] = np.pad(imgs[i][0], ((0, amt), (0, 0)), mode='constant')[amt:, :]

            # 随机清零最大5*5的区域
            x_size = np.random.randint(1, 6)
            y_size = np.random.randint(1, 6)
            x_begin = np.random.randint(0, 28 - x_size + 1)
            y_begin = np.random.randint(0, 28 - y_size + 1)
            imgs[i][0][x_begin:x_begin + x_size, y_begin:y_begin + y_size] = 0

    def whirl(self, img: np.ndarray, channels, high, width):
        im_rotated = np.einsum("ijk,lk->ijl", img, rotation_matrix(np.pi))
        # 利用爱因斯坦求和约定做矩阵乘法，实际上是将每个RGB像素点表示的三维空间点绕X轴（即红色通道轴）旋转180°。
        im2 = undo_normalise(im_rotated)
        self.showGrayscale(img)
