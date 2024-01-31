#include <iostream>
#include "src/constants.h"

int main() {
  auto board = case1_cheat_board();
  print_board(board);
  auto board_plus = case1_cheat_board_plus();
  print_board(board_plus);
  return 0;
}