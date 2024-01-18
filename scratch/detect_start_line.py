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

    return np.array([(x_min, x_min_y), (x_max, x_max_y), (x_y_min, y_min), (x_y_max, y_max)])


def fit_rectangle(corners):
    sorted_indices = np.lexsort((corners[:, 0], corners[:, 1]))
    corners = corners[sorted_indices]
    a, b, c, d = corners

    print((a+b+c+d)[0]/4, (a+b+c+d)[1]/4)
    virtual_c = np.array([(a+b+c+d)[0]/4, (a+b+c+d)[1]/4])
    ab = b - a
    bc = c - b
    cd = d - c
    da = a - d
    
    print(ab, bc, cd, da)

    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    norm_cd = np.linalg.norm(cd)
    norm_da = np.linalg.norm(da)

    print(norm_ab, norm_bc, norm_cd, norm_da)

    vec_norm = np.array([norm_ab, norm_bc, norm_cd, norm_da])
    vec_side = np.array([ab, bc, cd, da])
    # find the biggest vector, which might be the width side of the rectangle
    idx = np.argmax(vec_norm)
    w_side = vec_side[idx]
    print(w_side)
    w_side_p = np.array([-w_side[1], w_side[0]]) if w_side[1] >= 0 else np.array([w_side[1], -w_side[0]])

    unit_w_side_p = w_side_p / np.linalg.norm(w_side_p)
    return virtual_c, unit_w_side_p


def calcualte_theta(unit_v):
    x_axis = np.array([0, 1])
    theta = np.arccos((unit_v@x_axis) / (np.linalg.norm(unit_v)*np.linalg.norm(x_axis)))
    return theta



if __name__ == "__main__":
    # test_img = cv2.imread(
    #     "/home/william/Codes/find-landmark/data/start_line/fake/start_0001.jpg",
    #     cv2.IMREAD_GRAYSCALE,
    # )
    basedir = "/home/william/Codes/find-landmark/data/start_line/fake"
    all_file_names = [f for f in os.listdir(basedir)]
    i = 1
    for name in all_file_names:
        test_img = cv2.imread(os.path.join(basedir, name), cv2.IMREAD_GRAYSCALE)
        res = get_rectangle_vertices_simple(test_img)
        # print(res)
        c, unit_v = fit_rectangle(res)
        
        end_point = unit_v * 200

        img_copy = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
        for p in res:
            x = p[0]
            y = p[1]
            cv2.circle(img_copy, (y, x), 1, (128, 127, 255), 1)
        cv2.circle(img_copy, (int(c[1]), int(c[0])), 1, (0, 0, 255), 1)
        cv2.line(img_copy, (int(c[1]), int(c[0])), (int(c[1] + end_point[1]), int(c[0] + end_point[0])), (128, 127, 255), 1)
        cv2.imshow(f"vertex_{i}", img_copy)
        print(f">>> showing {i}th image.")
        print(f">>> unit vector is {unit_v}")
        theta = calcualte_theta(unit_v)
        print(f"theta is {theta} radians.")
        angle = np.degrees(theta)
        print(f"angle is {angle} degrees.")
        i+=1
    cv2.waitKey(0)
    cv2.destroyAllWindows()