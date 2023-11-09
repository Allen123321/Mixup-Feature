import torchvision.datasets as datasets
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize
from skimage.feature import hog
import cv2
import torch
from PIL import Image
import scipy.signal as signal  # 导入sicpy的signal模块


def get_mixup_hog_sobel_label(img, lamda=0.8):
    img = img.transpose((1, 2, 0))
    imgs_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    # x方向的Sobel算子
    operator_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
    # y方向的Sobel算子
    operator_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

    # 利用signal的convolve计算卷积

    sobel_x = signal.convolve2d(imgs_gray, operator_x, mode="same")
    sobel_y = signal.convolve2d(imgs_gray, operator_y, mode="same")
    hog_feature, hog_image = hog(imgs_gray, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                                 visualize=True)
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    # 对边缘强度进行归一化
    sobel /= sobel.max()
    mixup_feature = lamda * sobel + (1 - lamda) * hog_image

    return mixup_feature


# def get_mixup_lable(datas,lamda=0.8):
#     mixup_lable = []
#     for data in datas:
#         mixup_label.append(get_mixup_hog_sobel_label(image, lamda=lamda))

#     return np.array(mixup_label)


class MySTL10(datasets.STL10):
    def __init__(self, *args, lamda=0.5, **kwargs):
        super(MySTL10, self).__init__(*args, **kwargs)
        self.lamda = lamda

    def __getitem__(self, index):
        # 获取图像和标签
        img, target = self.data[index], self.labels[index]

        # 对图像进行预处理
        mixup_label = get_mixup_hog_sobel_label(img, lamda=self.lamda)

        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
            mixup_label = self.transform(mixup_label)

            # 返回图像和标签
        return img, mixup_label


if __name__ == '__main__':
    my_stl10 = MySTL10(root='./data', split='train', lamda=0.8, download=True)
    # 获取第一个数据样本
    my_stl10[0]







