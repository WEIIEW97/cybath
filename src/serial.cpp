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

C_EXPORT ortPathSegGPU* initialize_gpu() {
  ortPathSegGPU* ort_pathseg_gpu = new ortPathSegGPU();
  return ort_pathseg_gpu;
}

C_EXPORT SIG serial_start_line_detect(const cv::Mat& frame,
                                      ortPathSegGPU* stream) {
  cv::Mat start_line_mask;
  auto res = stream->processMask(frame, start_line_mask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    std::cerr << "processing segment error."
              << "\n";
  }

  if (start_line_mask.channels() == 3) {
    cv::cvtColor(start_line_mask, start_line_mask, cv::COLOR_BGR2GRAY);
  }

  auto vertices = get_rectangle_vertices(start_line_mask);
#if DEBUG
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
#if DEBUG
  std::cout << "center is " << c << std::endl;
#endif
  auto theta = calculate_theta(unit_v, v2);
  auto angle = rad2deg(theta);
#if DEBUG
  std::cout << ">>> unit vector is:  " << unit_v << "\n";
  std::cout << ">>> theta is: " << theta << "\n";
  std::cout << ">>> angle is: " << angle << "\n";
  std::cout << ">>> direction flag is: " << flag << "\n";
#endif
  SIG start_line_signal;
  start_line_signal.angle = angle;
  flag = (angle < 1.0f) ? PositionFlag::align : flag;
  start_line_signal.sign = flag;
  return start_line_signal;
}

C_EXPORT bool whether_to_begin_construction(const SIG& signal) {
  return true ? (signal.angle < 1.0f) : false;
}

C_EXPORT void delete_gpu(ortPathSegGPU* GPU) { delete (GPU); }