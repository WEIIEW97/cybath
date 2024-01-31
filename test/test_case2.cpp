#include <iostream>
#include <filesystem>
// #include <experimental/filesystem>
#include <chrono>
#include "../src/serial.h"
#include "../src/constants.h"
#include "../src/cnnseg/task2route.h"
#include "../src/cnnseg/task3tabletnavi.h"
#include "../src/cnnseg/task4emptyseat.h"

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
float camera_angle = -45;
const float indicate_thr1 = 0.85f;
const float indicate_thr2 = 0.55f;
const float indicate_thr3 = 0.55f;

/// TODO: optimize this class call, avoid call to construct in every loop
Footpath footpath(intrinsics, distortion_coeffs, camera_angle);

int extractNumber(const std::string& filename) {
  std::size_t pos = filename.find("rgb_");
  if (pos != std::string::npos) {
    pos += 4; // "rgb_" has 4 characters, so we move the position past it
    std::size_t start = pos;
    // Find the end of the number
    while (pos < filename.length() && std::isdigit(filename[pos])) {
      pos++;
    }
    return std::stoi(filename.substr(start, pos - start));
  }
  return -1; // Return -1 or an appropriate value if "rgb_" is not found
}

bool customSort(const std::string& a, const std::string& b) {
  return extractNumber(a) < extractNumber(b);
}

int main(int, char**) {
  using namespace std::chrono;
  auto last_time = high_resolution_clock::now();
//  std::string root_path = "/home/nvp/data/VIS/footpath_test3_data/color";
//  std::string depth_path = "/home/nvp/data/VIS/footpath_test3_data/aligned_depth_to_color";
    std::string root_path = "/home/william/data/cybathlon/weiwei0130/rgb";
    std::string depth_path = "/home/william/data/cybathlon/weiwei0130/depth";


//  auto gpu_carrier = initialize_gpu(road_onnx_model_path, line_onnx_model_path);
  auto multi_label_masks = make_shared<MultiLabelMaskSet>();

  auto gpu_carrier_case2 = new RouteTask(road_model_path.c_str(), border_model_path.c_str(), line_model_path.c_str());

  vector<string> all_image_paths;
  for (const auto& entry : fs::directory_iterator(root_path)) {
    all_image_paths.emplace_back(entry.path());
  }

  vector<string> all_depth_paths;
  for (const auto& entry : fs::directory_iterator(depth_path)) {
    all_depth_paths.emplace_back(entry.path());
  }

//  sort(all_image_paths.begin(), all_image_paths.end(),
//       [](const string& a, const string& b) { return a < b; });
//  sort(all_depth_paths.begin(), all_depth_paths.end(),
//       [](const string& a, const string& b) { return a < b; });

  sort(all_image_paths.begin(), all_image_paths.end(), customSort);
  sort(all_depth_paths.begin(), all_depth_paths.end(), customSort);

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
    cv::imshow("rgb input", rgb);
    cv::imshow("v mask", multi_label_masks->shape_v_lane);
    cv::imshow("gap mask", multi_label_masks->gap_lane);
    cv::imshow("road mask", multi_label_masks->road_lane);

    // auto start_line_flag = serial_start_line_detect(multi_label_masks);
    auto case2_package = serial_center_line_detect(multi_label_masks, footpath, depth, indicate_thr1, indicate_thr2, indicate_thr3);
    duration<double> elapsed_seconds = curr_time - last_time;
    double fps = 1.0 / elapsed_seconds.count();
    auto step_up_sign = case2_package.step_up_sign ? "step up!" : "not ready";
    auto step_down_sign = case2_package.step_down_sign ? "step down!" : "not ready";
    auto mind_gap_sign = case2_package.mind_gap ? "watch out!" : "not ready";

    cv::imshow("depth", depth);
    if (cv::waitKey(5) >= 0) {
      break; // Exit the loop if a key is pressed
    }
    cout << "begin frame: " << i << "\n";
    if (!case2_package.data.empty()) {
      cout << all_image_paths[i] << "\n";
      cout << case2_package.data[0] << "\n";
      cout << step_up_sign << "\n";
      cout << step_down_sign << "\n";
      cout << mind_gap_sign << "\n";
    }
    // cout << start_line_flag.angle << "\n";
    // cout << start_line_flag.sign << "\n";
    cout << "========================="
         << "\n";
    cout << "Current FPS: " << fps << "\n";
    flush(cout);
    last_time = curr_time;
  }
  cv::destroyAllWindows();

  delete(gpu_carrier_case2);
//  auto cheat_board = case1_cheat_board();
//  print_board(cheat_board);
  return 0;
}
