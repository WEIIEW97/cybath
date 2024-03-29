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

#ifndef FIND_LANDMARK_ESTIMATE_H
#define FIND_LANDMARK_ESTIMATE_H

#include <opencv2/opencv.hpp>

std::vector<cv::Point2f> estimate_trajectory(const cv::Mat& pgm, int thr,
                                             int kernel_size);
cv::Point get_wavefront_exit(const cv::Mat& pgm, const cv::Point& start_point,
                             int thr, int kernel_size);

#endif // FIND_LANDMARK_ESTIMATE_H
