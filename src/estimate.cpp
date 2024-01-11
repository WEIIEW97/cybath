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
#include <numeric>
#include "estimate.h"
#include "connected_component.h"
#include "wavefront.h"

std::vector<cv::Point> row_searching_reduce_method(const cv::Mat& binary_mask) {
  std::vector<cv::Point> middle_lane_coords;
  int h = binary_mask.rows;

  middle_lane_coords.reserve(h);

#pragma omp parallel for
  for (int i = 0; i < h; i++) {
    auto row = binary_mask.row(i);
    auto count = cv::countNonZero(row);

    if (count > 0) {
      cv::Mat locations;
      cv::findNonZero(row, locations);
      double sum = std::accumulate(
          locations.begin<cv::Point>(), locations.end<cv::Point>(), 0.0,
          [](double s, const cv::Point& p) { return s + p.x; });
      double mean = sum / count;
      middle_lane_coords.emplace_back(static_cast<int>(mean), i);
    }
  }
  return middle_lane_coords;
}

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