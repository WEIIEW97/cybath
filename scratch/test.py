import numpy as np
import cv2

LABEL_SCALAR = 50

if __name__ == "__main__":
    test_onnx_seg_path = "/home/william/extdisk/data/footpath_test3_data/seg/1704955330.584353_result.bmp"
    onnx_seg = cv2.imread(test_onnx_seg_path, cv2.IMREAD_GRAYSCALE)

    gap_mask = np.uint8(np.where(onnx_seg == 4*50, 1, 0) * 255)
    gap_indices = np.where(onnx_seg == 4 * 50)
    print(gap_indices)
    print(np.mean(gap_indices[0]))
    print(np.mean(gap_indices[1]))
    cv2.circle(gap_mask, [int(np.mean(gap_indices[1])), int(np.mean(gap_indices[0]))], 1, (0, 0, 0), 1)
    # print(global_start_end_lane.shape)
    cv2.imshow("global start end lane", gap_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()