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
#include "detect_center_line.h"

std::vector<cv::Point2f>
row_searching_reduce_method(const cv::Mat& binary_mask) {
  std::vector<cv::Point2f> middle_lane_coords;
  int h = binary_mask.rows;

  middle_lane_coords.reserve(h);

/// disable omp for causing the order vague
// #pragma omp parallel for
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
      middle_lane_coords.emplace_back(mean, i);
    }
  }

  std::reverse(middle_lane_coords.begin(), middle_lane_coords.end());
  return middle_lane_coords;
}

cv::Point
find_gap_centorid(const std::shared_ptr<MultiLabelMaskSet>& label_masks) {
  auto gap_mask = label_masks->gap_lane;

  std::vector<cv::Point> gap_indices;
  cv::findNonZero(gap_mask, gap_indices);
  cv::Point centorid = {-1, -1};
  if (!gap_indices.empty()) {
    cv::Point sum =
        std::accumulate(gap_indices.begin(), gap_indices.end(), cv::Point(0, 0),
                        [](const cv::Point& a, const cv::Point& b) {
                          return cv::Point(a.x + b.x, a.y + b.y);
                        });

    cv::Point c(static_cast<int>(sum.x / gap_indices.size()),
                static_cast<int>(sum.y / gap_indices.size()));
    centorid = c;
  }
  return centorid;
}

cv::Point find_gap_centorid(const cv::Mat& gap_mask) {
  std::vector<cv::Point> gap_indices;
  cv::findNonZero(gap_mask, gap_indices);
  cv::Point centorid = {-1, -1};
  if (!gap_indices.empty()) {
    cv::Point sum =
        std::accumulate(gap_indices.begin(), gap_indices.end(), cv::Point(0, 0),
                        [](const cv::Point& a, const cv::Point& b) {
                          return cv::Point(a.x + b.x, a.y + b.y);
                        });

    cv::Point c(static_cast<int>(sum.x / gap_indices.size()),
                static_cast<int>(sum.y / gap_indices.size()));
    centorid = c;
  }
  return centorid;
}

cv::Point find_gap_centorid(const std::vector<cv::Point>& gap_coords) {
  cv::Point centorid = {-1, -1};
  if (!gap_coords.empty()) {
    cv::Point sum =
        std::accumulate(gap_coords.begin(), gap_coords.end(), cv::Point(0, 0),
                        [](const cv::Point& a, const cv::Point& b) {
                          return cv::Point(a.x + b.x, a.y + b.y);
                        });

    cv::Point c(static_cast<int>(sum.x / gap_coords.size()),
                static_cast<int>(sum.y / gap_coords.size()));
    centorid = c;
  }
  return centorid;
}

bool is_mask_appear(const cv::Mat& gap_mask) {
  return cv::countNonZero(gap_mask) > 0;
}