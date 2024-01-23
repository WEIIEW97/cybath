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

ortPathSegGPU* initialize_gpu(const std::string& road_onnx_model_path,
                              const std::string& line_onnx_model_path) {
  auto* ort_pathseg_gpu =
      new ortPathSegGPU(road_onnx_model_path, line_onnx_model_path);
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

void get_labeled_masks_from_onnx(
    const cv::Mat& onnx_seg_result,
    std::shared_ptr<MultiLabelMaskSet>& multi_label_masks) {

  // split multi-labeled tasks into different masks;
  multi_label_masks->global_start_end_lane =
      (onnx_seg_result == 1 * LABEL_SCALAR);
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

std::vector<cv::Vec3d>
serial_center_line_detect(std::shared_ptr<MultiLabelMaskSet>& label_masks,
                          const cv::Mat& correspond_depth) {
  auto mask = label_masks->road_lane;
  /// temporally set parameters to be fixed.
  cv::Matx33d intrinsics{381.7600630735221,
                         0,
                         319.3731939266522,
                         0,
                         381.9814634837562,
                         243.68503537756743,
                         0,
                         0,
                         1};
  cv::Vec4d distortion_coeffs{-0.04442904360733734, 0.037326194718717384,
                              7.758816931839537e-06, 0.0005847117569966644};
  float camera_angle = -70;
  ///
  Footpath footpath(intrinsics, distortion_coeffs, camera_angle);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  auto control_poses =
      footpath.FollowPath(correspond_depth, mask, label_masks->gap_lane);
  return control_poses;
}

bool whether_to_begin_construction(const SIG& signal) {
  return true ? (signal.angle < 1.0f) : false;
}

void delete_gpu(ortPathSegGPU* GPU) { delete (GPU); }