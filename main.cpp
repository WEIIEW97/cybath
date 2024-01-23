#include <iostream>
#include <filesystem>
#include "src/exitpoint/connected_component.h"
#include "src/serial.h"

using namespace std;
namespace fs = std::filesystem;

int main(int, char**) {
  std::string root_path = "/home/nvp/data/VIS/footpath_test3_data/color";
  std::string depth_path =
      "/home/nvp/data/VIS/footpath_test3_data/aligned_depth_to_color";
  //  std::string root_path =
  //  "/home/william/extdisk/data/footpath_test3_data/color";

  //  std::string road_onnx_model_path =
  //  "/home/william/Codes/cybath/models/end2end_ocrnet_road_border.onnx";
  //  std::string line_onnx_model_path =
  //  "/home/william/Codes/cybath/models/end2end_ocrnet_line.onnx";
  std::string road_onnx_model_path =
      "/home/nvp/codes/cybath/models/end2end_ocrnet_road_border.onnx";
  std::string line_onnx_model_path =
      "/home/nvp/codes/cybath/models/end2end_ocrnet_line.onnx";

  auto gpu_carrier = initialize_gpu(road_onnx_model_path, line_onnx_model_path);
  int i = 0;
  auto multi_label_masks = make_shared<MultiLabelMaskSet>();

  vector<string> all_image_paths;
  for (const auto& entry : fs::directory_iterator(root_path)) {
    all_image_paths.emplace_back(entry.path());
  }

  vector<string> all_depth_paths;
  for (const auto& entry : fs::directory_iterator(depth_path)) {
    all_depth_paths.emplace_back(entry.path());
  }

  sort(all_image_paths.begin(), all_image_paths.end(),
       [](const string& a, const string& b) { return a < b; });
  sort(all_depth_paths.begin(), all_depth_paths.end(),
       [](const string& a, const string& b) { return a < b; });

  // for (const auto& p : all_image_paths) {
  //   cv::Mat rgb = cv::imread(p);
  //   cv::Mat path_seg = onnx_path_seg(rgb, gpu_carrier);
  //   get_labeled_masks_from_onnx(path_seg, multi_label_masks);
  //   // auto start_line_flag = serial_start_line_detect(multi_label_masks);
  //   cout << "begin frame: " << i << "\n";
  //   // cout << start_line_flag.angle << "\n";
  //   // cout << start_line_flag.sign << "\n";
  //   cout << "=========================" << "\n";
  //   i++;
  // }

  for (int i = 0; i < all_image_paths.size(); ++i) {
    cv::Mat rgb = cv::imread(all_image_paths[i]);
    cv::Mat depth = cv::imread(all_depth_paths[i], cv::IMREAD_ANYDEPTH);
    cv::Mat path_seg = onnx_path_seg(rgb, gpu_carrier);
    get_labeled_masks_from_onnx(path_seg, multi_label_masks);
    // auto start_line_flag = serial_start_line_detect(multi_label_masks);
    auto control_poses = serial_center_line_detect(multi_label_masks, depth);
    cout << "begin frame: " << i << "\n";
    cout << control_poses[0] << "\n";
    cout << control_poses.back() << "\n";
    // cout << start_line_flag.angle << "\n";
    // cout << start_line_flag.sign << "\n";
    cout << "========================="
         << "\n";
  }

  delete_gpu(gpu_carrier);
  return 0;
}
