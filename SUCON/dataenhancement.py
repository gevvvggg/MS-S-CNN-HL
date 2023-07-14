import numpy as np
from PIL import Image
import random

import numpy as np
from scipy.ndimage import rotate


def data_augmentation(images):
    """
    对输入的极化SAR图像数据进行数据增强：随机翻转、旋转、遮盖。

    Parameters:
        images: numpy array, 形状为(batch_size, img_size, img_size, 6)的极化SAR图像数据。

    Returns:
        numpy array, 形状为(batch_size, img_size, img_size, 6)的增强后的极化SAR图像数据。
    """
    images = np.transpose(images, (0, 2, 3, 1))
    batch_size, img_size, _, _ = images.shape

    # 随机翻转
    for i in range(batch_size):
        if np.random.rand() < 0.5:
            images[i] = np.flip(images[i], axis=0)
        if np.random.rand() < 0.5:
            images[i] = np.flip(images[i], axis=1)

    # 随机旋转
    for i in range(batch_size):
        angle = np.random.randint(-180, 180)
        images[i] = rotate(images[i], angle, axes=(0, 1), reshape=False, order=0, mode='reflect', cval=0.0)

    # 随机遮盖
    for i in range(batch_size):
        mask = np.ones((img_size, img_size, 6))
        x, y = np.random.randint(0, img_size, size=2)
        w, h = np.random.randint(img_size // 4, img_size // 2, size=2)
        mask[x - w // 2:x + w // 2, y - h // 2:y + h // 2, :] = 0
        images[i] *= mask
    images = np.transpose(images, (0, 3, 1, 2))
    return images


# batch_size = 6  # 批次大小
# img_size = 9  # 图像大小
# aug_images = data_augmentation(images, batch_size, img_size)

import numpy as np

# n_samples = 100  # 样本数
# batch_size = 6  # 批次大小
# img_size = 9  # 图像大小
# images = np.random.rand(n_samples, 6, img_size, img_size)  # 生成随机数据
# print(images.shape)
# # 将输入数据转换为(batch_size, img_size, img_size, 6)的格式
# n_batches = n_samples // batch_size
# images = images[:n_batches * batch_size]  # 抛弃多余的样本
# print(images.shape)

# print(images.shape)
# images = np.transpose(images, (0, 2, 3, 1))  # 转换为(batch_size, img_size, img_size, 6)的格式
# print(images.shape)
images=np.load(r'D:\sardata\jiujinshan\百分之10\trainfeature.npy')
images=data_augmentation(images)
np.save(r'D:\sardata\jiujinshan\百分之10\2augtrainfeature.npy',images)
# images = np.transpose(images, (0, 3, 1, 2))
# print(images.shape)
np.save(r'D:\sardata\jiujinshan\百分之10\2augtrainfeature.npy',images)
