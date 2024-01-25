// Projection factor of project to one frame depth camera
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#include "footpath.h"

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <random>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>

#include "loess/loess.h"
#include "detect_center_line.h"

Footpath::Footpath(const cv::Matx33d& intrinsics,
                   const cv::Vec4d& distortion_coeffs, float angel)

    : intrinsics_(intrinsics), distortion_coeffs_(distortion_coeffs),
      angle_(angel) {
//  R_cam_body_ = ypr2R(cv::Vec3d(90, 0 - 90)).t();
}

Footpath::~Footpath() = default;

std::vector<cv::Vec3d> Footpath::FollowPath(const cv::Mat& path_depth,
                                            const cv::Mat& path_mask,
                                            const cv::Mat& gap_mask) {
  std::vector<cv::Vec3d> control_poses;

  std::vector<cv::Point2f> path_middle_lane_coords;
  path_middle_lane_coords = RowSearchingReduceMethod(path_mask);
  if (path_middle_lane_coords.size() < 10)
    return control_poses;

  std::vector<cv::Point2f> fitting_coords;
  Smooth(path_middle_lane_coords, fitting_coords);
  fitting_coords.swap(path_middle_lane_coords);

  // Get the control points index
  auto control_points_index =
      GetControlPointsIndex(path_middle_lane_coords, gap_mask);
  if (control_points_index.empty())
    return control_poses;

  // Get the control point and control pose
  control_poses.reserve(control_points_index.size());
  std::vector<cv::Point2f> undist_path_middle_lane_coords;
  cv::undistortPoints(path_middle_lane_coords, undist_path_middle_lane_coords,
                      intrinsics_, distortion_coeffs_, cv::Mat(),
                      cv::Matx33d::eye());
  for (const auto& control_point_index : control_points_index) {
    // Get the control point
    float depth = path_depth.at<uint16_t>(
                      path_middle_lane_coords.at(control_point_index).x,
                      path_middle_lane_coords.at(control_point_index).y) /
                  1000.f;
    if (depth < 0.3)
      continue;
    cv::Vec3f control_point =
        cv::Vec3f(undist_path_middle_lane_coords.at(control_point_index).x,
                  undist_path_middle_lane_coords.at(control_point_index).y, 1) *
        depth;

    cv::Matx33d R = ypr2R(cv::Vec3f(0, 0, angle_));
    cv::Vec3f tvec = R * control_point;

    cv::Vec3f next_pos;
    next_pos(1) = tvec(2);
    next_pos(2) = -tvec(0);
    cv::Vec2f ori = normalize(cv::Vec2f(next_pos(1), next_pos(2)));
    float yaw = std::acos(ori.dot(cv::Vec2f(1, 0))) * 180 / M_PI;
    if (ori(1) < 0)
      yaw = -yaw;
    next_pos(0) = yaw;
    control_poses.push_back(next_pos);
#ifdef DEBUG
    std::cout << ">>>position: " << next_pos.t() << std::endl;
#endif
  }
  return control_poses;
}

std::vector<cv::Point2f>
Footpath::RowSearchingReduceMethod(const cv::Mat& img_mask) {
  return row_searching_reduce_method(img_mask);
}

cv::Vec3f
Footpath::GetIntersectPointFromLP(const cv::Vec3f& plane_normal_vector,
                                  const float plane_intercept,
                                  const cv::Vec3f& point) {
  cv::Vec3f line_normal_vector = cv::normalize(point);
  cv::Vec3f result =
      point - (plane_normal_vector.dot(point) + plane_intercept) /
                  (plane_normal_vector.dot(line_normal_vector)) *
                  line_normal_vector;

  return result;
}

void Footpath::Smooth(const std::vector<cv::Point2f>& input,
                      std::vector<cv::Point2f>& output) {
  std::vector<LOESS::Point> inpoints, outpoints;
  std::vector<double> valsout;

  outpoints.resize(input.size(), LOESS::Point(1, 0));
  for (int i = 0; i < input.size(); ++i) {
    LOESS::Point tmp_point(1, 0);
    tmp_point.val(input.at(i).x);
    tmp_point[0] = input.at(i).y;
    inpoints.push_back(tmp_point);

    outpoints[i][0] = input.at(i).y;
  }

  loess(inpoints, outpoints, valsout, 0.05, 0, 2, 8);

  output.reserve(input.size());
  for (int i = 0; i < input.size(); ++i) {
    output.emplace_back(valsout.at(i), input.at(i).y);
  }
}

std::vector<int> Footpath::GetControlPointsIndex(
    const std::vector<cv::Point2f>& path_middle_lane, const cv::Mat& gap_mask,
    int step) {

  std::vector<int> result_index;
  cv::Point2f gap_mass_center = {0, 0};
  auto is_gap_detected = cv::countNonZero(gap_mask);
  if (is_gap_detected > 0) {
    gap_mass_center = find_gap_centorid(gap_mask);
  }
  result_index.reserve(
      (path_middle_lane.size() / step) +
      10); // for safe & redundancy capacity, delete 1 point and add 2 points
#pragma omp parallel for
  for (int i = step; i < path_middle_lane.size() - 1; i += step) {

    /// TODO: optimize ths strategy
//    auto l2_dist = (is_gap_detected > 0)
//                       ? cv::norm(path_middle_lane[i] - gap_mass_center)
//                       : (step + 1);
//    if (l2_dist < step) {
//      // if selected point was around of gap_mass_center, then choose the
//      // step/2 near points
//      result_index.emplace_back(i - step / 2);
//      result_index.emplace_back(i + step / 2);
//    } else {
//      result_index.emplace_back(i);
//    }
    result_index.emplace_back(i);
  }
  return result_index;
}

cv::Vec3d Footpath::Rvec2ypr(const cv::Vec3d& rvec) {
  cv::Matx33d R;
  cv::Rodrigues(rvec, R);

  auto n = R.col(0);
  auto o = R.col(1);
  auto a = R.col(2);

  cv::Vec3d ypr;
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r =
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}

cv::Vec3d Footpath::R2ypr(const cv::Matx33d& R) {
  auto n = R.col(0);
  auto o = R.col(1);
  auto a = R.col(2);

  cv::Vec3d ypr;
  double y = atan2(n(1), n(0));
  double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  double r =
      atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}

cv::Matx33d Footpath::ypr2R(cv::Vec3d ypr) {
  double y = ypr(0) / 180.0 * M_PI;
  double p = ypr(1) / 180.0 * M_PI;
  double r = ypr(2) / 180.0 * M_PI;

  cv::Matx33d Rz(cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1);

  cv::Matx33d Ry(cos(p), 0., sin(p), 0., 1., 0., -sin(p), 0., cos(p));

  cv::Matx33d Rx(1., 0., 0., 0., cos(r), -sin(r), 0., sin(r), cos(r));

  return Rz * Ry * Rx;
}
