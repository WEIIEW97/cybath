// Projection factor of project to one frame depth camera
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <queue>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "footpath.h"

void LoadImages(const std::string& data_path, std::queue<double>& times_color,
                std::queue<std::string>& img_color,
                std::queue<double>& times_depth,
                std::queue<std::string>& img_depth);

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: footpath [data_path]." << std::endl;
    return 1;
  }

  std::string data_path(argv[1]);
  if (data_path.back() != '/')
    data_path += "/";

  std::queue<double> times_color, times_depth;
  std::queue<std::string> imgs_color, imgs_depth;
  LoadImages(data_path, times_color, imgs_color, times_depth, imgs_depth);

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

  Footpath footpath(intrinsics, distortion_coeffs, -70, true);

  int cnt = 0;

  while (true) {
    std::string img_color_name, img_depth_name, img_mask_name;
    double time_color, time_depth;
    if (!times_color.empty() && !times_depth.empty()) {
      time_color = times_color.front();
      time_depth = times_depth.front();

      if (time_color < time_depth - 0.003) {
        times_color.pop();
        imgs_color.pop();
        std::cout << "throw color image." << std::endl;
        continue;
      } else if (time_color > time_depth + 0.003) {
        times_depth.pop();
        imgs_depth.pop();
        std::cout << "throw depth image." << std::endl;
        continue;
      } else {
        img_color_name = data_path + imgs_color.front();
        img_depth_name = data_path + imgs_depth.front();
        std::string::size_type ns = img_color_name.rfind('/');
        //        std::string sub_name = img_color_name.substr(ns + 1);
        //        img_mask_name = data_path + "sam_seg_mask_close/sam_seg_" +
        //        sub_name;
        std::string::size_type ne = img_color_name.rfind('.');
        std::string sub_name = img_color_name.substr(ns + 1, ne - ns - 1);
        img_mask_name = data_path + "color_0102/" + sub_name + "_result.jpg";
        cv::Mat img_color = cv::imread(img_color_name, -1);
        cv::Mat img_depth = cv::imread(img_depth_name, -1);
        cv::Mat img_mask = cv::imread(img_mask_name, -1);
        if (img_color.empty() || img_depth.empty() || img_mask.empty()) {
          std::cerr << "#ERROR: image is empty." << std::endl;
          break;
        }

        cv::Mat kernel =
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(img_mask, img_mask, cv::MORPH_OPEN, kernel);

        auto control_poses =
            footpath.FollowPath(img_depth, img_mask, img_color);

        times_color.pop();
        imgs_color.pop();
        times_depth.pop();
        imgs_depth.pop();
        ++cnt;
      }
    } else {
      break;
    }
  }

  std::cout << "count: " << cnt << std::endl;

  return 0;
}

void LoadImages(const std::string& data_path, std::queue<double>& times_color,
                std::queue<std::string>& img_color,
                std::queue<double>& times_depth,
                std::queue<std::string>& img_depth) {
  auto Load = [](const std::string& file_name, std::queue<double>& times,
                 std::queue<std::string>& img) {
    std::ifstream fin;
    fin.open(file_name, std::ios::in);
    if (!fin.is_open())
      std::cerr << "#ERROR: cannot open the file." << std::endl;
    while (!fin.eof()) {
      std::string s;
      std::getline(fin, s);
      std::stringstream ss(s);
      double timestamp;
      std::string img_name;
      ss >> timestamp;
      ss >> img_name;
      times.push(timestamp);
      img.push(img_name);
    }
  };

  std::string color_file_name = data_path + "color.txt";
  std::string depth_file_name = data_path + "aligned_depth_to_color.txt";

  Load(color_file_name, times_color, img_color);
  Load(depth_file_name, times_depth, img_depth);
}
