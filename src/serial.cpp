/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "serial.h"
#define LABEL_SCALAR 50

ortPathSegGPU* initialize_gpu() {
  auto* ort_pathseg_gpu = new ortPathSegGPU();
  return ort_pathseg_gpu;
}

cv::Mat onnx_path_seg(const cv::Mat& frame, ortPathSegGPU* stream) {
  cv::Mat onnx_seg;
  auto res = stream->processMask(frame, onnx_seg);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    std::cerr << "processing segment error."
              << "\n";
  }
  return onnx_seg;
}

void get_labeled_masks_from_onnx(const cv::Mat& onnx_seg_result, std::shared_ptr<MultiLabelMaskSet>& multi_label_masks) {

  // split multi-labeled tasks into different masks;
  multi_label_masks->global_start_end_lane = (onnx_seg_result == 1 * LABEL_SCALAR);
  multi_label_masks->border_lane = (onnx_seg_result == 2 * LABEL_SCALAR);
  multi_label_masks->shape_v_lane = (onnx_seg_result == 3 * LABEL_SCALAR);
  multi_label_masks->gap_lane = (onnx_seg_result == 4 * LABEL_SCALAR);
  multi_label_masks->road_lane = (onnx_seg_result == 5 * LABEL_SCALAR);

  multi_label_masks->global_start_end_lane *= 255;
  multi_label_masks->border_lane *= 255;
  multi_label_masks->shape_v_lane *= 255;
  multi_label_masks->gap_lane *= 255;
  multi_label_masks->road_lane *= 255;
}

SIG serial_start_line_detect(std::shared_ptr<MultiLabelMaskSet>& label_masks) {
  auto vertices = get_rectangle_vertices(label_masks->global_start_end_lane);
  SIG start_line_signal;
  if (vertices.size() == 1) {
    return start_line_signal;
  }
#ifdef DEBUG
  std::cout << "vertices is: "
            << "\n";
  for (auto& p : vertices) {
    std::cout << p << " ";
  }
  std::cout << "\n";
#endif
  cv::Point2f c, unit_v;
  PositionFlag flag;
  fit_rectangle(vertices, c, unit_v, flag);
  cv::Point2f v2 = {-1, 0};
#ifdef DEBUG
  std::cout << "center is " << c << std::endl;
#endif
  auto theta = calculate_theta(unit_v, v2);
  auto angle = rad2deg(theta);
#ifdef DEBUG
  std::cout << ">>> unit vector is:  " << unit_v << "\n";
  std::cout << ">>> theta is: " << theta << "\n";
  std::cout << ">>> angle is: " << angle << "\n";
  std::cout << ">>> direction flag is: " << flag << "\n";
#endif

  start_line_signal.angle = angle;
  flag = (angle < 1.0f) ? PositionFlag::align : flag;
  start_line_signal.sign = flag;
  return start_line_signal;
}

bool whether_to_begin_construction(const SIG& signal) {
  return true ? (signal.angle < 1.0f) : false;
}

void delete_gpu(ortPathSegGPU* GPU) { delete (GPU); }