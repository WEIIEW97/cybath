import numpy as np
import cv2

LABEL_SCALAR = 50

if __name__ == "__main__":
    test_onnx_seg_path = "/home/william/extdisk/data/footpath_test3_data/seg/1704955326.415914_result.jpg"
    onnx_seg = cv2.imread(test_onnx_seg_path, cv2.IMREAD_GRAYSCALE)
    m0 = np.where(onnx_seg == 0*LABEL_SCALAR, 1, 0)
    m1 = np.where(onnx_seg == 1*LABEL_SCALAR, 1, 0)
    m2 = np.where(onnx_seg == 2*LABEL_SCALAR, 1, 0)
    m3 = np.where(onnx_seg == 3*LABEL_SCALAR, 1, 0)
    m4 = np.where(onnx_seg == 4*LABEL_SCALAR, 1, 0)
    m5 = np.where(onnx_seg == 5*LABEL_SCALAR, 1, 0)

    # m = m1 + m2 + m3 + m4 + m5
    # m = np.uint8(m) * 255

    m1 = np.uint8(m5) * 255
    # global_start_end_lane *= 255
    # global_start_end_lane = np.uint8(global_start_end_lane)
    # print(onnx_seg.shape)
    # print(global_start_end_lane.shape)
    cv2.imshow("global start end lane", m1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()