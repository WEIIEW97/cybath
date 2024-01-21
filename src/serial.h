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

#include "detect_start_line.h"
#include "../cnnseg/ortpathseggpu.h"

#define C_EXPORT

struct SIG {
  float angle = -1.0f;
  PositionFlag sign = PositionFlag::error;
};

C_EXPORT ortPathSegGPU* initialize_gpu();
C_EXPORT SIG serial_start_line_detect(const cv::Mat& frame,
                                      ortPathSegGPU* stream);
C_EXPORT bool whether_to_begin_construction(const SIG& signal);
C_EXPORT void delete_gpu(ortPathSegGPU* GPU);