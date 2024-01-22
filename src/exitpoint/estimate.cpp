/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "estimate.h"
#include "connected_component.h"
#include "wavefront.h"
#include "../centorid/detect_center_line.h"

std::vector<cv::Point> estimate_trajectory(const cv::Mat& pgm, int thr,
                                           int kernel_size) {
  ConnectedComponent cc(pgm, thr);
  cc.cross_rectangle_conv(kernel_size);
  cv::Mat walkable = cc.find_largest_connected_component();
  auto trajectory = row_searching_reduce_method(walkable);
  return trajectory;
}

cv::Point get_wavefront_exit(const cv::Mat& pgm, const cv::Point& start_point,
                             int thr, int kernel_size) {
  WaveFront wf(pgm, start_point, thr);
  wf.cross_rectangle_conv(kernel_size);
  cv::Mat path_map = wf.explore(WaveFrontDirection().way8, kernel_size);
  return wf.find_most_likely_exit_point(path_map);
}