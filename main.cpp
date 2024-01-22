#include <iostream>
#include <filesystem>
#include "src/exitpoint/connected_component.h"
#include "src/exitpoint/estimate.h"
#include "src/startline/detect_start_line.h"

using namespace std;
namespace fs = std::filesystem;

int main(int, char**) {
  //    string pgm_path = "/home/william/Codes/cybath/test.pgm";
  //    auto pgm = cv::imread(pgm_path, -1);
  //    auto exit_point1 = estimate_trajectory(pgm, 254, 15).back();
  //    auto exit_point2 = get_wavefront_exit(pgm, cv::Point(31, 5), 254, 15);
  //    cout << exit_point1 << ", " << exit_point2 << endl;
  //    cv::Mat pgm_flip;
  //    cv::flip(pgm, pgm_flip, 0);
  ////    for(auto& p : exit_point) {
  //      cv::circle(pgm_flip, exit_point1, 2, cv::Scalar(125, 0, 0), 2);
  //      cv::circle(pgm_flip, exit_point2, 2, cv::Scalar(77, 0, 0), 2);
  ////    }
  //    cv::imshow("end point", pgm_flip);
  //    cv::waitKey(0);
  //    cv::destroyAllWindows();

  // string rootdir = "/home/william/codes/cybath/data/start_line/fake";
  // vector<string> all_file_names;

  // for (const auto& f : fs::directory_iterator(rootdir)) {
  //   all_file_names.push_back(f.path().filename().string());
  // }

  // int i = 1;
  // cv::Point2f v2 = {-1, 0};
  // for (const auto& name : all_file_names) {
  //   string full_path = rootdir + "/" + name;
  //   cv::Mat frame = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
  //   auto res = get_rectangle_vertices(frame);
  //   cout << "vertices is " << endl;
  //   for (auto& p : res) {
  //     cout << p << " ";
  //   }
  //   cout << endl;
  //   cv::Point2f c, unit_v;
  //   PositionFlag flag;
  //   fit_rectangle(res, c, unit_v, flag);

  //   cout << "center is: " << c << endl;

  //   cv::Point end_point = unit_v * 200;
  //   cv::Mat frame_rgb;
  //   cv::cvtColor(frame, frame_rgb, cv::COLOR_GRAY2BGR);
  //   for (auto& p : res) {
  //     cv::swap(p.x, p.y);
  //     cv::circle(frame_rgb, p, 1, (128, 127, 255), 1);
  //   }
  //   cv::circle(frame_rgb, cv::Point(int(c.y), int(c.x)), 1, (0, 0, 255), 1);
  //   cv::line(frame_rgb, cv::Point(int(c.y), int(c.x)),
  //            cv::Point(int(c.y + end_point.y), int(c.x + end_point.x)),
  //            (128, 127, 255), 1);

  //   std::stringstream ss;
  //   ss << "vertex_" << i;
  //   std::string win_name = ss.str();

  //   cv::imshow(win_name, frame_rgb);
  //   cout << ">>> showing " << i << "th image" << endl;
  //   cout << ">>> unit vector is:  " << unit_v << endl;
  //   auto theta = calculate_theta(unit_v, v2);
  //   cout << ">>> theta is: " << theta << endl;
  //   cout << ">>> angle is: " << rad2deg(theta) << endl;
  //   cout << ">>> direction flag is: " << flag << endl;
  //   i++;
  // }
  // cv::waitKey(0);
  // cv::destroyAllWindows();
  cout << fs::current_path().relative_path() << endl;
  return 0;
}
