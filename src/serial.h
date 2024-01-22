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

#pragma once

#include "startline/detect_start_line.h"
#include "cnnseg/ortpathseggpu.h"
#include <memory>

struct SIG {
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

ortPathSegGPU* initialize_gpu();
std::shared_ptr<MultiLabelMaskSet>
get_labeled_masks_from_onnx(const cv::Mat& onnx_seg_result);
SIG serial_start_line_detect(const std::shared_ptr<MultiLabelMaskSet>& label_masks);
bool whether_to_begin_construction(const std::shared_ptr<MultiLabelMaskSet>& label_masks);
void delete_gpu(ortPathSegGPU* GPU);