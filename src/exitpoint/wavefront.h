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

#ifndef FIND_LANDMARK_WAVEFRONT_H
#define FIND_LANDMARK_WAVEFRONT_H

#include <opencv2/opencv.hpp>
#include <utility>
#include <vector>

using std::vector;

struct WaveFrontDirection {
  WaveFrontDirection() = default;
  vector<vector<int>> way4 = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
  vector<vector<int>> way8 = {{0, 1}, {1, 0},  {0, -1}, {-1, 0},
                              {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
  vector<vector<int>> way6 = {{0, 1},  {1, 0}, {0, -1},
                              {-1, 0}, {1, 1}, {1, -1}};
};

class WaveFront {
public:
  cv::Mat img_;
  cv::Point start_point_;
  int thr_;
  cv::Mat walkable_area_;

public:
  WaveFront(const cv::Mat& img, cv::Point start_point, int thr) {
    cv::flip(img, img_, 0);
    start_point_ = std::move(start_point);
    thr_ = thr;
  }
  void cross_rectangle_conv(int kernel_size);
  cv::Mat explore(const vector<vector<int>>& direction, int kernel_size);
  cv::Point find_most_likely_exit_point(const cv::Mat& path_map);
};

#endif // FIND_LANDMARK_WAVEFRONT_H
