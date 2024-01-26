#include <iostream>
#include <filesystem>
#include <chrono>
#include "src/exitpoint/connected_component.h"
#include "src/serial.h"
#include "src/constants.h"
#include "src/cnnseg/task2route.h"

using namespace std;
namespace fs = std::filesystem;

/// temporally set parameters to be fixed.
cv::Matx33d intrinsics{381.7600630735221,
                       0,
                       319.3731939266522,
                       0,
                       381.9814634837562,
                       243.68503537756743,
                       0,
                       0,
                       1};
cv::Vec4d distortion_coeffs{-0.04442904360733734, 0.037326194718717384,
                            7.758816931839537e-06, 0.0005847117569966644};
float camera_angle = -40;
/// TODO: optimize this class call, avoid call to construct in every loop
Footpath footpath(intrinsics, distortion_coeffs, camera_angle);

int main(int, char**) {
  using namespace std::chrono;
  auto last_time = high_resolution_clock::now();
//  std::string root_path = "/home/nvp/data/VIS/footpath_test3_data/color";
//  std::string depth_path =
      "/home/nvp/data/VIS/footpath_test3_data/aligned_depth_to_color";
    std::string root_path = "/home/william/extdisk/data/footpath_test3_data/color";
    std::string depth_path = "/home/william/extdisk/data/footpath_test3_data/aligned_depth_to_color";

    std::string road_onnx_model_path =
    "/home/william/Codes/cybath/models/end2end_ocrnet_road_border.onnx";
    std::string line_onnx_model_path =
    "/home/william/Codes/cybath/models/end2end_ocrnet_line.onnx";
//  std::string road_onnx_model_path =
      "/home/nvp/codes/cybath/models/end2end_ocrnet_road_border.onnx";
//  std::string line_onnx_model_path =
      "/home/nvp/codes/cybath/models/end2end_ocrnet_line.onnx";

//  auto gpu_carrier = initialize_gpu(road_onnx_model_path, line_onnx_model_path);
  auto multi_label_masks = make_shared<MultiLabelMaskSet>();

  auto gpu_carrier_case2 = new RouteTask(road_model_path, border_model_path, line_model_path);

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

  const float indicate_thr = 0.3;
  cv::Mat path_seg;
  bool has_shoe = false;
  cv::Mat visimg;
  std::vector<int> shoe_pos;
  for (int i = 0; i < all_image_paths.size(); ++i) {
    auto curr_time = high_resolution_clock::now();
    cv::Mat rgb = cv::imread(all_image_paths[i]);
    cv::Mat depth = cv::imread(all_depth_paths[i], cv::IMREAD_ANYDEPTH);
//    cv::Mat path_seg = onnx_path_seg(rgb, gpu_carrier);
    gpu_carrier_case2->processMask(rgb, path_seg, has_shoe, shoe_pos, visimg);
    get_labeled_masks_from_onnx(path_seg, multi_label_masks);
    // auto start_line_flag = serial_start_line_detect(multi_label_masks);
    auto case2_package = serial_center_line_detect(multi_label_masks, footpath, depth, indicate_thr);
    duration<double> elapsed_seconds = curr_time - last_time;
    double fps = 1.0 / elapsed_seconds.count();
    auto step_up_sign = case2_package.step_up_sign ? "step up!" : "not ready";
    auto step_down_sign = case2_package.step_down_sign ? "step down!" : "not ready";

    cout << "begin frame: " << i << "\n";
    cout << case2_package.data[0] << "\n";
    cout << step_up_sign  << "\n";
    cout << step_down_sign << "\n";
    // cout << start_line_flag.angle << "\n";
    // cout << start_line_flag.sign << "\n";
    cout << "========================="
         << "\n";
    cout << "Current FPS: " << fps << "\r";
    flush(cout);
    last_time = curr_time;
  }

  delete(gpu_carrier_case2);
//  auto cheat_board = case1_cheat_board();
//  print_board(cheat_board);
  return 0;
}
