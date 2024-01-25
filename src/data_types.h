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

#ifndef CYBATHLON_DATA_TYPES_H
#define CYBATHLON_DATA_TYPES_H

#include <opencv2/opencv.hpp>

enum PositionFlag {
  error = -1,
  align = 0,
  turn_left = 1,
  turn_right = 2,
  stop = 3,
};

struct Case1Package {
  float angle = -1.0f;
  PositionFlag sign = PositionFlag::error;
};

struct MultiLabelMaskSet {
  cv::Mat global_start_end_lane;
  cv::Mat border_lane;
  cv::Mat shape_v_lane;
  cv::Mat gap_lane;
  cv::Mat road_lane;
};

struct Case2Package {
  bool ready_for_step_up = false;
  bool ready_for_step_down = false;
  std::vector<cv::Vec3d> data;
};

struct Case3Package {
  float depth;
  PositionFlag sign;
};

#endif // CYBATHLON_DATA_TYPES_H
