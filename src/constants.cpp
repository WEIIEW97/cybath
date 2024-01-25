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
#include "constants.h"

void print_board(const std::unordered_map<int, std::pair<int, int>>& board) {
  for (const auto& pair : board) {
    std::cout << "Index: " << pair.first << " -> (" << pair.second.first << ", "
              << pair.second.second << ")\n";
  }
}

std::unordered_map<int, std::pair<int, int>> case1_cheat_board() {
  std::unordered_map<int, std::pair<int, int>> labeled_cheat_board;
  const int size = 5;
  const int begin_board_offset = 2;
  const int initial_offset_x = 1610 - 1475;
  const int initial_offset_y = 1060 + 1200;
  const int dx = 550;
  const int dy = 550;

  // Create a 5x5 board of std::vector<std::pair<int, int>>
  std::vector<std::vector<std::pair<int, int>>> board(
      size, std::vector<std::pair<int, int>>(size));

  for (int i = 4; i >= 0; --i) {
    if (i == 4) {
      board[i][begin_board_offset] =
          std::make_pair(initial_offset_x, initial_offset_y);
    } else {
      board[i][begin_board_offset].first =
          board[i + 1][begin_board_offset].first;
      board[i][begin_board_offset].second =
          board[i + 1][begin_board_offset].second + dy;
    }

    for (int j = 1; j < 3; ++j) {
      board[i][begin_board_offset + j].first =
          board[i][begin_board_offset + j - 1].first + dx;
      board[i][begin_board_offset + j].second =
          board[i][begin_board_offset + j - 1].second;

      board[i][begin_board_offset - j].first =
          board[i][begin_board_offset - j + 1].first - dx;
      board[i][begin_board_offset - j].second =
          board[i][begin_board_offset - j + 1].second;
    }
  }

  int begin_index = 25;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      labeled_cheat_board[begin_index] = board[i][j];
      begin_index--;
    }
  }
  return labeled_cheat_board;
}