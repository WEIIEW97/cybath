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

#ifndef CYBATHLON_DEPTH_NAVIGATION_H
#define CYBATHLON_DEPTH_NAVIGATION_H

#include "../data_types.h"
#include <opencv2/opencv.hpp>
#include <memory>

Case3Package navigate_by_depth_and_box_3(bool has_cabinet,
                                         std::vector<int>& cabin_pos,
                                         bool has_tab,
                                         std::vector<int>& tab_pos,
                                         const cv::Mat& aligned_depth);
Case3Package navigate_by_depth_and_box_4(bool has_cabinet,
                                         std::vector<int>& position,
                                         const cv::Mat& rgb,
                                         const cv::Mat& aligned_depth,
                                         float dist_1, float dist_2);
#endif // CYBATHLON_DEPTH_NAVIGATION_H