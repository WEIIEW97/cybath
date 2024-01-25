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

#ifndef CYBATHLON_CONSTANTS_H
#define CYBATHLON_CONSTANTS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>

void print_board(const std::unordered_map<int, std::pair<int, int>>& board);
std::unordered_map<int, std::pair<int, int>> case1_cheat_board();

#endif // CYBATHLON_CONSTANTS_H
