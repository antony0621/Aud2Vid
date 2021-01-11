import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def conv2d(img, kernel):
    print(np.shape(img))
    img = torch.Tensor(img)
    img = torch.transpose(img, 0, 2).unsqueeze(0)
    kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0).repeat([3, 3, 1, 1])
    out = F.conv2d(img, kernel, padding=1).squeeze()
    out = out.numpy()
    print(np.shape(out))
    out = np.transpose(out, (1, 2, 0))
    out = np.rot90(out, -1)
    out = np.fliplr(out)
    return out


if __name__ == '__main__':
    image = cv.imread("000069.jpg")
    g, b, r = cv.split(image)
    image = cv.merge([r, g, b])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    kernel = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]) / 9
    diff = conv2d(image, kernel)
    print(diff[:, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(diff[:, :, 0])
    plt.show()


