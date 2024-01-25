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
#include "depth_navigation.h"

Case3Package navigate_by_depth_and_box_3(bool has_cabinet, std::vector<int>& cabin_pos, bool has_tab, std::vector<int>& tab_pos, const cv::Mat& aligned_depth, float thr) {
  Case3Package msg{};

  int half_w = aligned_depth.cols / 2;

  float global_mean_dpeth;
  float mean_cabinet_depth = 0.f;
  float mean_tab_depth = 0.f;

  if (has_cabinet) {
    int cabin_trunc_x = (cabin_pos[2] - cabin_pos[0]) / 4;
    int cabin_trunc_y = (cabin_pos[3] - cabin_pos[1]) / 4;
    cv::Point upper_left = {cabin_pos[0] + cabin_trunc_x, cabin_pos[1] + cabin_trunc_y};
    cv::Point bottom_right = {cabin_pos[2] - cabin_trunc_x, cabin_pos[3] - cabin_trunc_y};
    cv::Rect roi(upper_left, bottom_right);
    mean_cabinet_depth = cv::mean(aligned_depth(roi))[0];
  }

  if (has_tab) {
    int tab_trunc_x = (tab_pos[2] - tab_pos[0]) / 4;
    int tab_trunc_y = (tab_pos[3] - tab_pos[1]) / 4;
    cv::Point upper_left = {tab_pos[0] + tab_trunc_x, tab_pos[1] + tab_trunc_y};
    cv::Point bottom_right = {tab_pos[2] - tab_trunc_x, tab_pos[3] - tab_trunc_y};
    cv::Rect roi(upper_left, bottom_right);
    mean_cabinet_depth = cv::mean(aligned_depth(roi))[0];
  }

  global_mean_dpeth = (mean_cabinet_depth + mean_tab_depth) / 2;
  msg.depth = global_mean_dpeth;
  /// very brutal and simple way to tell whether to turn left or right
  auto cabin_center_x = (cabin_pos[0] + cabin_pos[2]) / 2;
  auto tab_center_x = (tab_pos[0] + tab_pos[2]) / 2;
  auto bb_center_x = (cabin_center_x + tab_center_x) / 2;
  if (bb_center_x > half_w) {
    msg.sign = PositionFlag::turn_right;
  } else if (bb_center_x < half_w) {
    msg.sign = PositionFlag::turn_left;
  } else {
    msg.sign = PositionFlag::align;
  }

  if (global_mean_dpeth <= thr) {
    msg.sign = PositionFlag::stop;
  }

  return msg;
}

Case3Package navigate_by_depth_and_box_4(bool has_cabinet, std::vector<int>& position, const cv::Mat& rgb, const cv::Mat& aligned_depth) {
  Case3Package msg{};

  uint8_t black_thr = 50;

  if (has_cabinet) {
    cv::Point upper_left = {position[0], position[1]};
    cv::Point bottom_right = {position[2], position[3]};
    cv::Rect roi(upper_left, bottom_right);
    auto rgb_block = rgb(roi);

    cv::Scalar lb(0, 0, 0);
    cv::Scalar ub(black_thr, black_thr, black_thr);
    cv::Mat black_mask;
    cv::inRange(rgb_block, lb, ub, black_mask);

    /// take the black area mask;
    cv::mean(aligned_depth, black_mask)[0];
  }

}