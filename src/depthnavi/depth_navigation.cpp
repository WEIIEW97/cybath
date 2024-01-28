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

void very_sloppy_left_right_turn_indicator(int center_x, int pos_x, int tolerance_thr,
                                           PositionFlag& flag) {
  if (pos_x > center_x + tolerance_thr) {
    flag = PositionFlag::turn_right;
  } else if (pos_x < center_x - tolerance_thr) {
    flag = PositionFlag::turn_left;
  } else {
    flag = PositionFlag::align;
  }
}

