#include <iostream>
#include <filesystem>
#include "src/exitpoint/connected_component.h"
#include "src/exitpoint/estimate.h"
#include "src/startline/detect_start_line.h"
#include "src/serial.h"

using namespace std;
namespace fs = std::filesystem;

int main(int, char**) {
  std::string root_path = "/home/william/extdisk/data/footpath_test3_data/statr_line_test";
  std::string image_path = "/home/william/extdisk/data/footpath_test3_data/statr_line_test/1704955326.415914.png";

  cv::Mat rgb = cv::imread(image_path);

  auto gpu_carrier = initialize_gpu();
  cv::Mat path_seg = onnx_path_seg(rgb, gpu_carrier);
  auto multi_masks = get_labeled_masks_from_onnx(path_seg);

  auto start_line_flag = serial_start_line_detect(multi_masks);
  cout << start_line_flag.angle << endl;
  cout << start_line_flag.sign << endl;

  delete_gpu(gpu_carrier);
  return 0;
}
