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

#ifndef CYBATHLON_DETECT_CENTER_LINE_H
#define CYBATHLON_DETECT_CENTER_LINE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

#include "../data_types.h"

std::vector<cv::Point2f>
row_searching_reduce_method(const cv::Mat& binary_mask);
cv::Point
find_gap_centorid(const std::shared_ptr<MultiLabelMaskSet>& label_masks);
cv::Point find_gap_centorid(const cv::Mat& gap_mask);
cv::Point find_gap_centorid(const std::vector<cv::Point>& gap_coords);
bool is_mask_appear(const cv::Mat& gap_mask);

#endif // CYBATHLON_DETECT_CENTER_LINE_H
