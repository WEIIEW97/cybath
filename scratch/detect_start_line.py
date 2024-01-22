import numpy as np
import cv2
import os


def get_rectangle_vertices_simple(mask):
    indices = np.where(mask >= 127)
    x_vec = indices[0]
    y_vec = indices[1]

    x_min = x_vec.min()
    x_max = x_vec.max()

    y_min = y_vec.min()
    y_max = y_vec.max()

    x_min_y = y_vec[np.where(x_vec == x_min)][0]
    x_max_y = y_vec[np.where(x_vec == x_max)][-1]

    x_y_min = x_vec[np.where(y_vec == y_min)][0]
    x_y_max = x_vec[np.where(y_vec == y_max)][-1]

    return np.array(
        [(x_min, x_min_y), (x_max, x_max_y), (x_y_min, y_min), (x_y_max, y_max)]
    )


def distinguish_position(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray):
    ### 0 means error
    ### 1 means turn left
    ### 2 means turn right
    flag = 0
    # if a.all() == b.all() and c.all() == d.all():
    #     return flag
    if a.all() != b.all():
        s1 = a
        s2 = b
    else:
        s1 = c
        s2 = d

    if s1[1] <= s2[1]:
        if s1[0] >= s2[0]:
            flag = 1
        else:
            flag = 2
    return flag


def fit_rectangle(corners):
    sorted_indices = np.lexsort((corners[:, 0], corners[:, 1]))
    corners = corners[sorted_indices]
    a, b, c, d = corners

    flag_map = {"error": 0, "turn_left": 1, "turn_right": 2}
    position = distinguish_position(a, b, c, d)

    def get_key_by_value(flag_map, value):
        for k, v in flag_map.items():
            if v == value:
                return k
        return None

    print(get_key_by_value(flag_map, position))

    virtual_c = np.array([(a + b + c + d)[0] / 4, (a + b + c + d)[1] / 4])
    if position == 2:
        v1 = a - b
        v2 = c - a
        v3 = d - c
        v4 = b - d
    elif position == 1:
        v1 = b - a
        v2 = d - b
        v3 = c - d
        v4 = a - c
    else:
        raise ValueError("Cannot detect four vertices.")

    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
    # norm_v3 = np.linalg.norm(v3)
    # norm_v4 = np.linalg.norm(v4)

    # vec_norm = np.array([norm_v1, norm_v2, norm_v3, norm_v4])
    # vec_side = np.array([v1, v2, v3, v4])
    ## find the biggest vector, which might be the width side of the rectangle
    ## use v2 as the baseline
    # idx = np.argmax(vec_norm)
    w_side = v2
    w_side_p = (
        np.array([-w_side[1], w_side[0]])
        if w_side[1] >= 0
        else np.array([w_side[1], -w_side[0]])
    )

    unit_w_side_p = w_side_p / np.linalg.norm(w_side_p)
    return virtual_c, unit_w_side_p


def calculate_theta(unit_v, v2):
    theta = np.arccos((unit_v @ v2) / (np.linalg.norm(unit_v) * np.linalg.norm(v2)))
    return theta


if __name__ == "__main__":
    test_img = cv2.imread(
        "/home/william/Codes/cybath/data/start_line/fake/start_0004.jpg",
        cv2.IMREAD_GRAYSCALE,
    )
    res = get_rectangle_vertices_simple(test_img)
    c, unit_v = fit_rectangle(res)
    print(f">>> unit vector is {unit_v}")
    v2 = np.array([-1, 0])
    theta = calculate_theta(unit_v, v2)
    print(f"theta is {theta} radians.")
    angle = np.degrees(theta)
    print(f"angle is {angle} degrees.")
    # basedir = "/home/william/Codes/cybath/data/start_line/fake"
    # all_file_names = [f for f in os.listdir(basedir)]
    # i = 1

    # v2 = np.array([-1, 0])
    # for name in all_file_names:
    #     test_img = cv2.imread(os.path.join(basedir, name), cv2.IMREAD_GRAYSCALE)
    #     res = get_rectangle_vertices_simple(test_img)
    #     c, unit_v = fit_rectangle(res)

    #     # print("center is :", c)

    # #     end_point = unit_v * 200

    # #     img_copy = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    # #     for p in res:
    # #         x = p[0]
    # #         y = p[1]
    # #         cv2.circle(img_copy, (y, x), 1, (128, 127, 255), 1)
    # #     cv2.circle(img_copy, (int(c[1]), int(c[0])), 1, (0, 0, 255), 1)
    # #     cv2.line(
    # #         img_copy,
    # #         (int(c[1]), int(c[0])),
    # #         (int(c[1] + end_point[1]), int(c[0] + end_point[0])),
    # #         (128, 127, 255),
    # #         1,
    # #     )
    # #     cv2.imshow(f"vertex_{i}", img_copy)
    #     print(f">>> showing {i}th image.")
    #     print(f">>> unit vector is {unit_v}")
    #     theta = calculate_theta(unit_v, v2)
    #     print(f"theta is {theta} radians.")
    #     angle = np.degrees(theta)
    #     print(f"angle is {angle} degrees.")
    #     i += 1
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
