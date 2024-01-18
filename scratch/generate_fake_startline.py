import numpy as np
import cv2
import os

ROOT_PATH = os.getcwd()
SAVE_PATH = os.path.join(ROOT_PATH, "data/start_line/fake_py")
os.makedirs(SAVE_PATH, exist_ok=True)


def generate_fake_startline_impl(cx, cy, h, w, angle, bg_h, bg_w, scale):
    nh = np.arange(0, bg_h * scale, scale)
    nw = np.arange(0, bg_w * scale, scale)

    coord = np.meshgrid(nh, nw, indexing="ij")
    y = coord[0]
    x = coord[1]

    x = x - cx
    y = y - cy

    theta = np.deg2rad(angle)
    y_rot = y * np.cos(theta) - x * np.sin(theta)
    x_rot = y * np.sin(theta) + x * np.cos(theta)

    # mask = np.abs(x_rot) <= w/2 & np.abs(y_rot) <= h/2
    mask = np.bitwise_and(np.abs(y_rot) <= h / 2, np.abs(x_rot) <= w / 2)
    return mask


def generate_fake_startline(iters, h, w, angle_range, bg_h, bg_w, scale):
    for i in range(iters):
        cx = np.random.randint(bg_w / 2 - 30, bg_w / 2 + 30)
        cy = np.random.randint(bg_h - 60, bg_h - 10)
        angle = np.random.randint(angle_range[0], angle_range[1])
        mask = generate_fake_startline_impl(cx, cy, h, w, angle, bg_h, bg_w, scale)
        ext_name = "start_{:04d}.jpg".format(i+1)
        mask = np.uint8(mask) * 255
        cv2.imwrite(os.path.join(SAVE_PATH, ext_name), mask)


if __name__ == "__main__":
    h = 10
    w = 600
    scale = 1
    angle_range = (-15, 15)
    bg_h = 480
    bg_w = 640

    n_iters = 10
    generate_fake_startline(n_iters, h, w, angle_range, bg_h, bg_w, scale)
    print("done!")
