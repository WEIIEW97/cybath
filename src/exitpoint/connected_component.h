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

#ifndef FIND_LANDMARK_CONNECTED_COMPONENT_H
#define FIND_LANDMARK_CONNECTED_COMPONENT_H
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

class ConnectedComponent {
public:
  int thr_ = 0;
  cv::Mat walkable_area_;
  cv::Mat img_;

public:
  explicit ConnectedComponent(const cv::Mat& img, int thr) {
    cv::flip(img, img_, 0);
    thr_ = thr;
  }
  void cross_rectangle_conv(int kernel_size);
  void set_threshold(int thr);
  cv::Mat find_largest_connected_component();
};

#endif // FIND_LANDMARK_CONNECTED_COMPONENT_H
