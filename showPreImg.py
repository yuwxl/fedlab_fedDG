import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def get_coutour_sample(y_true):
    # print("y_true shape",y_true.shape)
    disc_mask = np.expand_dims(y_true[..., 0], axis=2)

    disc_erosion = ndimage.binary_erosion(disc_mask[..., 0], iterations=1).astype(disc_mask.dtype)
    disc_dilation = ndimage.binary_dilation(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_contour = np.expand_dims(disc_mask[..., 0] - disc_erosion, axis = 2)
    disc_bg = np.expand_dims(disc_dilation - disc_mask[..., 0], axis = 2)
    cup_mask = np.expand_dims(y_true[..., 1], axis=2)

    cup_erosion = ndimage.binary_erosion(cup_mask[..., 0], iterations=1).astype(cup_mask.dtype)
    cup_dilation = ndimage.binary_dilation(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_contour = np.expand_dims(cup_mask[..., 0] - cup_erosion, axis = 2)
    cup_bg = np.expand_dims(cup_dilation - cup_mask[..., 0], axis = 2)

    return [disc_contour, disc_bg, cup_contour, cup_bg]


def draw_image(image):
    # print("in draw",image)
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.xticks([])
    plt.yticks([])

    return 0


if __name__ == '__main__':

    plt.figure(figsize=(6,6))
    img = np.load('../fed_output/result_fedlab/prediction/3_sample1.npy_img.npy')
    img = img.transpose((1, 2, 0))

    plt.subplot(3, 4, 2)
    draw_image(img)
    plt.xlabel("rawIMG", fontsize=10)

# 真实
    img_np2 = np.load('../fed_output/result_fedlab/prediction/3_sample1.npy_gth.npy')
    mask_patch = img_np2.transpose((1,2,0))
    disc_contour, disc_bg, cup_contour, cup_bg = get_coutour_sample(mask_patch)

    plt.subplot(3,4,5)
    draw_image(disc_contour)
    plt.xlabel("raw_disc_contour", fontsize=10)

    plt.subplot(3,4,6)
    draw_image(disc_bg)
    plt.xlabel("raw_disc_bg", fontsize=10)

    plt.subplot(3,4,7)
    draw_image(cup_contour)
    plt.xlabel("raw_cup_contour", fontsize=10)

    plt.subplot(3,4,8)
    draw_image(cup_bg)
    plt.xlabel("raw_cup_bg", fontsize=10)

# 预测
    img_np2 = np.load('../fed_output/result_fedlab/prediction/3_sample1.npy_pred.npy')
    mask_patch = img_np2.transpose((1,2,0))
    disc_contour, disc_bg, cup_contour, cup_bg = get_coutour_sample(mask_patch)

    plt.subplot(3, 4, 9)
    draw_image(disc_contour)
    plt.xlabel("pre_disc_contour", fontsize=10)

    plt.subplot(3, 4, 10)
    draw_image(disc_bg)
    plt.xlabel("pre_disc_bg", fontsize=10)

    plt.subplot(3, 4, 11)
    draw_image(cup_contour)
    plt.xlabel("pre_cup_contour", fontsize=10)

    plt.subplot(3, 4, 12)
    draw_image(cup_bg)
    plt.xlabel("pre_cup_bg", fontsize=10)
    plt.show()

