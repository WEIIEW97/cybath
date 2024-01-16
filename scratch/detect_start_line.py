import numpy as np
import cv2

def get_rectangle_vertices(mask):
    indices = np.where(mask == 255)
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

    return [(x_min, x_min_y), (x_max, x_max_y), (x_y_min, y_min), (x_y_max, y_max)]   





if __name__ == "__main__":
    test_img = cv2.imread("/home/william/Codes/find-landmark/data/start_line/fake/start_0010.jpg", cv2.IMREAD_GRAYSCALE)
    res = get_rectangle_vertices(test_img)
    print(res)
    img_copy = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
    for p in res:
        x = p[0]
        y = p[1]
        cv2.circle(img_copy, (y, x), 1, (128, 127, 255), 1)
    cv2.imshow("vertex", img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()