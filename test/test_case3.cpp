#include <iostream>
 #include <filesystem>
//#include <experimental/filesystem>
#include <chrono>
#include "../src/serial.h"
#include "../src/constants.h"
#include "../src/cnnseg/task3tabletnavi.h"

using namespace std;
namespace fs = std::filesystem;

#define PRINT_VECTOR_1D(x)                                                     \
  for (auto& v : x)                                                            \
    std::cout << "[" << v << ",]"                                              \
              << "\n";

int main(int, char**) {
  using namespace std::chrono;
  auto last_time = high_resolution_clock::now();
  //  std::string root_path = "/home/nvp/data/VIS/footpath_test3_data/color";
  //  std::string depth_path =
  //  "/home/nvp/data/VIS/footpath_test3_data/aligned_depth_to_color";
  std::string root_path = "/home/william/data/cybathlon/touchscreen_data/color";
  std::string depth_path =
      "/home/william/data/cybathlon/touchscreen_data/aligned_depth_to_color";

  //  auto gpu_carrier = initialize_gpu(road_onnx_model_path,
  //  line_onnx_model_path);
  auto multi_label_masks = make_shared<MultiLabelMaskSet>();

  auto gpu_carrier_case3 =
      new TabletnaviTask(tablet_model_path.c_str(), border_model_path.c_str());

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

  /// initialization shit

  int res;
  cv::Mat path_seg;
  bool avaliable[6] = {false};
  std::vector<int> cab_pos, tab_pos;

  bool has_cabinet = false;
  bool has_tablet = false;
  cv::Mat border_mask;
  cv::Mat visimg;
  /// initialization shit

  for (int i = 0; i < all_image_paths.size(); ++i) {
    auto curr_time = high_resolution_clock::now();
    cv::Mat rgb = cv::imread(all_image_paths[i]);
    cv::Mat depth = cv::imread(all_depth_paths[i], cv::IMREAD_ANYDEPTH);
    res = gpu_carrier_case3->findObject(rgb, has_cabinet, cab_pos, has_tablet,
                                        tab_pos, border_mask, visimg);
    cv::imshow("rgb image", rgb);
    if (cv::waitKey(30) >= 0) {
      break; // Exit the loop if a key is pressed
    }
    auto case3_package = serial_navigate_by_depth_and_box_3(
        has_cabinet, cab_pos, has_tablet, tab_pos, depth, 400);
    duration<double> elapsed_seconds = curr_time - last_time;
    double fps = 1.0 / elapsed_seconds.count();

    cout << "begin frame: " << i << "\n";
    cout << ">>> cabinet position is: "
         << "\n";
    PRINT_VECTOR_1D(cab_pos)
    cout << ">>> tablet postion is: "
         << "\n";
    PRINT_VECTOR_1D(tab_pos)
    cout << ">>> current distance is: " << case3_package.depth << endl;
    cout << ">>> current turning signal is: " << case3_package.sign << endl;
    cout << "========================="
         << "\n";
    cout << "Current FPS: " << fps << "\n";
    flush(cout);
    last_time = curr_time;
  }
  cv::destroyAllWindows();

  delete (gpu_carrier_case3);
  return 0;
}
