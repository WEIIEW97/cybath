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
#include "connected_component.h"
#include <vector>

void ConnectedComponent::set_threshold(int thr) { thr_ = thr; }

void ConnectedComponent::cross_rectangle_conv(int kernel_size) {
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

cv::Mat ConnectedComponent::find_largest_connected_component() {
  cv::Mat cc;
  cv::Mat walk;
  walkable_area_.convertTo(walk, CV_8UC1, 255.0);
  int num_labels = cv::connectedComponents(walk, cc);

  // find the largest connected component
  std::vector<int> area(num_labels, 0);
  for (int i = 0; i < cc.rows; ++i) {
    auto label_ptr = cc.ptr<int>(i);
    for (int j = 0; j < cc.cols; ++j) {
      area[label_ptr[j]]++;
    }
  }

  auto largest_iter = std::max_element(area.begin() + 1, area.end());
  int largest_label = std::distance(area.begin(), largest_iter);

  cv::Mat largest_component = cv::Mat::zeros(img_.size(), CV_8U);
  for (int i = 0; i < cc.rows; i++) {
    auto cc_ptr = cc.ptr<int>(i);
    auto lc_ptr = largest_component.ptr<uint8_t>(i);
    for (int j = 0; j < cc.cols; j++) {
      if (cc_ptr[j] == largest_label) {
        lc_ptr[j] = 255;
      }
    }
  }
  return largest_component;
}