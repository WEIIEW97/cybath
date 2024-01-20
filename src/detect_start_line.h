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

#ifndef FIND_LANDMARK_DETECT_START_LINE_H
#define FIND_LANDMARK_DETECT_START_LINE_H

#include <opencv2/opencv.hpp>
#include <vector>

enum PositionFlag {
  error = 0,
  turn_left = 1,
  turn_right = 2,
};

inline float rad2deg(float theta) { return theta * 180.0f / M_PI; }
std::vector<cv::Point> get_rectangle_vertices(const cv::Mat& mask);
PositionFlag distinguish_position(const cv::Point& a, const cv::Point& b,
                                  const cv::Point& c, const cv::Point& d);
void fit_rectangle(std::vector<cv::Point>& corners, cv::Point2f& virtual_c,
                   cv::Point2f& unit_w_side_p, PositionFlag& flag);
float calculate_theta(const cv::Point2f& unit_v, const cv::Point2f& v2);

#endif // FIND_LANDMARK_DETECT_START_LINE_H
