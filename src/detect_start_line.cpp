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
#include "detect_start_line.h"
#include <algorithm>
#include <cmath>
#include <optional>

template<typename T>
std::optional<size_t> find_last_index(const std::vector<T>& vec, T value) {
  std::optional<size_t> last_index = std::nullopt;
  for(auto i=0; i<vec.size(); ++i) {
    if (vec[i] == value) last_index = i;
  }
  return last_index;
}

std::vector<cv::Point> get_rectangle_vertices(const cv::Mat& mask) {
  std::vector<int> x_vec, y_vec;

  for (int i = 0; i < mask.rows; i++) {
    auto mask_ptr = mask.ptr<uchar>(i);
    for (int j = 0; j < mask.cols; j++) {
      if (mask_ptr[j] >= 127) {
        x_vec.push_back(i);
        y_vec.push_back(j);
      }
    }
  }

  auto x_min_max = std::minmax_element(x_vec.begin(), x_vec.end());
  int x_min = *x_min_max.first;
  int x_max = *x_min_max.second;

  auto y_min_max = std::minmax_element(y_vec.begin(), y_vec.end());
  int y_min = *y_min_max.first;
  int y_max = *y_min_max.second;

  auto x_min_y_it = std::find(x_vec.begin(), x_vec.end(), x_min);
  int x_min_y = y_vec[std::distance(x_vec.begin(), x_min_y_it)];

  auto x_max_y_it = find_last_index<int>(x_vec, x_max);
  int x_max_y = y_vec[x_max_y_it.value()];

  auto x_y_min_it = std::find(y_vec.begin(), y_vec.end(), y_min);
  int x_y_min = x_vec[std::distance(y_vec.begin(), x_y_min_it)];

  auto x_y_max_it = find_last_index<int>(y_vec, y_max);
  int x_y_max = x_vec[x_y_max_it.value()];

  std::vector<cv::Point> vertices = {
      cv::Point(x_min, x_min_y),
      cv::Point(x_max, x_max_y),
      cv::Point(x_y_min, y_min),
      cv::Point(x_y_max, y_max)
  };

  return vertices;
}

void fit_rectangle(std::vector<cv::Point>& corners, cv::Point2f& virtual_c, cv::Point2f& unit_w_side_p) {
  // Sort the corners
  std::sort(corners.begin(), corners.end(), [](const cv::Point& a, const cv::Point& b) {
    return (a.y < b.y) || (a.y == b.y && a.x < b.x);
  });

  cv::Point2f a = corners[0];
  cv::Point2f b = corners[1];
  cv::Point2f c = corners[2];
  cv::Point2f d = corners[3];

  // Calculate the virtual center
  virtual_c = (a + b + c + d) * 0.25;

  // Calculate the vectors for each side
  cv::Point2f ab = b - a;
  cv::Point2f bc = c - b;
  cv::Point2f cd = d - c;
  cv::Point2f da = a - d;

  // Calculate the norms (lengths) of each side
  float norm_ab = cv::norm(ab);
  float norm_bc = cv::norm(bc);
  float norm_cd = cv::norm(cd);
  float norm_da = cv::norm(da);

  std::vector<float> vec_norms = {norm_ab, norm_bc, norm_cd, norm_da};
  std::vector<cv::Point2f> vec_sides = {ab, bc, cd, da};

  // Find the biggest vector, which might be the width side of the rectangle
  auto max_it = std::max_element(vec_norms.begin(), vec_norms.end());
  int idx = std::distance(vec_norms.begin(), max_it);
  cv::Point2f w_side = vec_sides[idx];

  // Calculate perpendicular vector
  cv::Point2f w_side_p;
  if (w_side.y >= 0) {
    w_side_p = cv::Point2f(-w_side.y, w_side.x);
  } else {
    w_side_p = cv::Point2f(w_side.y, -w_side.x);
  }

  // Normalize the perpendicular vector
  unit_w_side_p = w_side_p * (1.0f / cv::norm(w_side_p));
}

float calculate_theta(const cv::Point2f& unit_v, const cv::Point2f& v2) {
  float dot_product = unit_v.x * v2.x + unit_v.y * v2.y;
  float norm_unit_v = std::sqrt(unit_v.x * unit_v.x + unit_v.y * unit_v.y);
  float norm_v2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

  float theta = std::acos(dot_product / (norm_unit_v * norm_v2));
  return theta;
}

