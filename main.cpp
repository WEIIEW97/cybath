#include <iostream>
#include <filesystem>
#include "src/exitpoint/connected_component.h"
#include "src/exitpoint/estimate.h"
#include "src/startline/detect_start_line.h"
#include "src/serial.h"

using namespace std;
namespace fs = std::filesystem;

int main(int, char**) {
  std::string root_path = "/home/nvp/data/VIS/footpath_test3_data/color";
  std::string image_path = "/home/nvp/data/VIS/footpath_test3_data/color/1704955326.415914.png";


  auto gpu_carrier = initialize_gpu();
  int i = 0;
  auto multi_label_masks = make_shared<MultiLabelMaskSet>();
  for (const auto& entry : fs::directory_iterator(root_path)) {
    auto image_name = entry.path();
    auto full_path = root_path + "/" + std::string(image_name);
    cout << full_path << "\n";
  cv::Mat rgb = cv::imread(image_path);

  cv::Mat path_seg = onnx_path_seg(rgb, gpu_carrier);
  get_labeled_masks_from_onnx(path_seg, multi_label_masks);

  auto start_line_flag = serial_start_line_detect(multi_label_masks);
  cout << "begin frame: " << i << "\n";
  cout << start_line_flag.angle << "\n";
  cout << start_line_flag.sign << "\n";
  cout << "=========================" << "\n";
  i++;
  }

  delete_gpu(gpu_carrier);
  return 0;
}
