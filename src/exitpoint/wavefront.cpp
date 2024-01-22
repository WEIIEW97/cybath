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
#include "wavefront.h"
#include <deque>

void WaveFront::cross_rectangle_conv(int kernel_size) {
  cv::Mat kernel = cv::Mat::ones(1, kernel_size, CV_32FC1);
  cv::Mat mask = img_ >= thr_;
  mask.convertTo(mask, CV_32F);
  mask /= 255.0f;

  cv::Mat horizontal_mask, vertical_mask;
  cv::filter2D(mask, horizontal_mask, -1, kernel);
  horizontal_mask = horizontal_mask == kernel_size;
  cv::filter2D(mask, vertical_mask, -1, kernel.t());
  vertical_mask = vertical_mask == kernel_size;

  walkable_area_ = horizontal_mask.mul(vertical_mask);
}

cv::Mat WaveFront::explore(const vector<vector<int>>& direction,
                           int kernal_size) {
  cross_rectangle_conv(kernal_size);
  int h = walkable_area_.rows;
  int w = walkable_area_.cols;
  walkable_area_ /= 255;
  walkable_area_.convertTo(walkable_area_, CV_32S);
  cv::Mat visited = cv::Mat::zeros(h, w, CV_32S);
  cv::Mat path_map = cv::Mat::ones(h, w, CV_32S) * -1;
  std::deque<cv::Point> q;
  q.push_back(start_point_);
  path_map.at<int>(start_point_.y, start_point_.x) = 0;

  while (!q.empty()) {
    auto [x, y] = q.front();
    q.pop_front();
    for (auto& D : direction) {
      int dy = D[0];
      int dx = D[1];
      int nx = x + dx;
      int ny = y + dy;
      if (ny >= 0 && ny < h && nx >= 0 && nx < w &&
          walkable_area_.at<int>(ny, nx) == 1 && !visited.at<int>(ny, nx)) {
        visited.at<int>(ny, nx) = 1;
        path_map.at<int>(ny, nx) = path_map.at<int>(y, x) + 1;
        q.emplace_back(nx, ny);
      }
    }
  }
  return path_map;
}

cv::Point WaveFront::find_most_likely_exit_point(const cv::Mat& path_map) {
  cv::Point minloc, maxloc;
  double min, max;
  cv::minMaxLoc(path_map, &min, &max, &minloc, &maxloc);
  return maxloc;
}