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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include "loess/loess.h"
#include "detect_center_line.h"

//cv::VideoWriter video_writer;

Footpath::Footpath(const cv::Matx33d& intrinsics,
                   const cv::Vec4d& distortion_coeffs,
                   float angel,
                   bool verbose)

    : intrinsics_(intrinsics),
      distortion_coeffs_(distortion_coeffs),
      angle_(angel),
      verbose_(verbose) {
  R_cam_body_ = ypr2R(cv::Vec3d(90, 0 -90)).t();

//  video_writer.open("./example.avi",
//                    cv::VideoWriter::fourcc('M','J','P','G'),
//                    30,
//                    cv::Size(640, 480));
}

Footpath::~Footpath() {
//  video_writer.release();
}

Footpath::ControlPoses Footpath::GetControlPoses(const cv::Mat& img_depth,
                                                 const cv::Mat& img_mask,
                                                 const cv::Mat& img_color) {
  ControlPoses control_poses;

  std::vector<cv::Point2f> path_middle_lane_coords;
  path_middle_lane_coords = RowSearchingReduceMethod(img_mask);
  if (path_middle_lane_coords.size() < 10)
    return control_poses;

  cv::Mat img_show;
  if (verbose_) {
    if (!img_color.empty())
      img_color.copyTo(img_show, img_mask);
    else
      cv::cvtColor(img_mask, img_show, cv::COLOR_GRAY2RGB);

    std::vector<cv::Point2f> fitting_coords;
    Smooth(path_middle_lane_coords, fitting_coords);
    fitting_coords.swap(path_middle_lane_coords);

    int cnt = 0;
    for (const auto &it : path_middle_lane_coords) {
      cv::circle(img_show, it, 1,
                 GetColor(cnt++, 0, path_middle_lane_coords.size()));
    }

//    for (const auto &it : fitting_coords) {
//      cv::circle(img_show, it, 1, cv::Scalar(0, 255, 0));
//    }

//    int num_points = path_middle_lane_coords.size();
//    std::vector<float> x, y;
//    for (const auto& item : path_middle_lane_coords) {
//      x.push_back(item.x);
//      y.push_back(item.y);
//      std::cout << item.x << ", " << item.y << "; ..." << std::endl;
//    }

    cv::imshow("test", img_show);
    cv::waitKey(0);
  }

  // Get all point coordinate of footpath mask
  std::vector<cv::Point> footpath_mask_coords;
  cv::findNonZero(img_mask, footpath_mask_coords);
  int num_footpath_mask_coords = footpath_mask_coords.size();

  // Randomly obtain N points for plane fitting
  int num_plane_fit = std::min(5000, num_footpath_mask_coords);
  std::vector<int> footpath_mask_index(num_footpath_mask_coords);
  std::iota(footpath_mask_index.begin(), footpath_mask_index.end(), 0);
  std::shuffle(footpath_mask_index.begin(), footpath_mask_index.end(),
               std::mt19937{ std::random_device{}() });
  std::vector<cv::Point2f> points_plane_fitting(num_plane_fit);
  for (int i = 0; i < num_plane_fit; ++i) {
    points_plane_fitting.at(i) =
        footpath_mask_coords.at(footpath_mask_index.at(i));
  }
  std::vector<cv::Point2f> undist_points_plane_fitting;
  cv::undistortPoints(points_plane_fitting, undist_points_plane_fitting,
                      intrinsics_, distortion_coeffs_,
                      cv::Mat(), cv::Matx33d::eye());
  pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (int i = 0; i < num_plane_fit; ++i) {
    float d = img_depth.at<uint16_t>(points_plane_fitting.at(i)) / 1000.f;
    pcl::PointXYZ point_xyz;
    point_xyz.x = undist_points_plane_fitting.at(i).x * d;
    point_xyz.y = undist_points_plane_fitting.at(i).y * d;
    point_xyz.z = d;
    point_cloud->emplace_back(point_xyz);
  }

  // RANSAC fitting plane
  pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(
      new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(point_cloud));
  pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
  ransac.setDistanceThreshold(0.025);
  ransac.setMaxIterations(30);
  ransac.setNumberOfThreads(4);
  ransac.computeModel();

  if (ransac.inliers_.size() < point_cloud->size() * 0.75) {
    std::cout << "#WARNING: fitting plane false." << std::endl;
    return control_poses;
  }

  Eigen::VectorXf coefficient;
  ransac.getModelCoefficients(coefficient);
  if (coefficient(3) < 0)
    coefficient *= -1;
  if (verbose_)
    std::cout << "plane function: "
              << coefficient(0) << "x + "
              << coefficient(1) << "y + "
              << coefficient(2) << "z + "
              << coefficient(3) << " = 0" << std::endl;

  const cv::Vec3f plane_normal_vector(coefficient(0),
                                      coefficient(1),
                                      coefficient(2));
  const float plane_intercept = coefficient(3);

  // Randomize the coordinates of 300 points on the plane
  int num_solve_pnp = std::min(300, num_plane_fit);
  cv::Mat points_3d(num_solve_pnp, 3, CV_32F);
  std::vector<cv::Point2f> points_2d;
  points_2d.reserve(num_solve_pnp);
  for (int i = 0; i < num_solve_pnp; ++i) {
    cv::Vec3f p3d = GetIntersectPointFromLP(
        plane_normal_vector, plane_intercept,
        cv::Vec3f(undist_points_plane_fitting.at(i).x,
                  undist_points_plane_fitting.at(i).y,
                  1));
    points_3d.at<float>(i, 0) = p3d(0);
    points_3d.at<float>(i, 1) = p3d(1);
    points_3d.at<float>(i, 2) = p3d(2);
    points_2d.emplace_back(points_plane_fitting.at(i));
  }

  // Get the control points index
  auto control_points_index =
      GetControlPointsIndex(path_middle_lane_coords, 45);
  if (control_points_index.empty())
    return control_poses;

  // Get the control point and control pose
  control_poses.reserve(control_points_index.size());
  std::vector<cv::Point2f> undist_path_middle_lane_coords;
  cv::undistortPoints(path_middle_lane_coords, undist_path_middle_lane_coords,
                      intrinsics_, distortion_coeffs_,
                      cv::Mat(), cv::Matx33d::eye());
  for (const auto& control_point_index : control_points_index) {
    int start = control_point_index - 2, end = control_point_index + 2;
    // Get the control point
    cv::Vec3f control_point = GetIntersectPointFromLP(
        plane_normal_vector, plane_intercept,
        cv::Vec3f(undist_path_middle_lane_coords.at(control_point_index).x,
                  undist_path_middle_lane_coords.at(control_point_index).y,
                  1));

    // Constructing a virtual coordinate system
    cv::Vec3f point_s = GetIntersectPointFromLP(
        plane_normal_vector, plane_intercept,
        cv::Vec3f(undist_path_middle_lane_coords.at(start).x,
                  undist_path_middle_lane_coords.at(start).y,
                  1));
    cv::Vec3f point_e = GetIntersectPointFromLP(
        plane_normal_vector, plane_intercept,
        cv::Vec3f(undist_path_middle_lane_coords.at(end).x,
                  undist_path_middle_lane_coords.at(end).y,
                  1));
    cv::Vec3f axis_x = cv::normalize(point_e - point_s);
    cv::Vec3f axis_y = plane_normal_vector.cross(axis_x);
    cv::Matx33f rot(
        axis_x(0), axis_x(1), axis_x(2),
        axis_y(0), axis_y(1), axis_y(2),
        plane_normal_vector(0), plane_normal_vector(1), plane_normal_vector(2));

    // Solve the control point pose
    cv::Mat point_center;
    cv::repeat(control_point.t(), num_solve_pnp, 1, point_center);
    cv::Mat ps3d = (rot * (points_3d - point_center).t()).t();
    cv::Vec3d rvec, tvec;
    cv::solvePnP(ps3d, points_2d, intrinsics_, distortion_coeffs_,
                 rvec, tvec, false, cv::SOLVEPNP_IPPE);

    if (verbose_)
      cv::drawFrameAxes(
          img_show, intrinsics_, distortion_coeffs_, rvec, tvec, 0.1, 2);

    cv::Matx33d R;
    cv::Rodrigues(rvec, R);

    // T_w_cam
//    R = R.t();
//    tvec = -R * tvec;
//    cv::Vec3d ypr = R2ypr(R);
//    control_poses.emplace_back(ypr, tvec);

    cv::Vec3d ypr = R2ypr(R);
    R = ypr2R(cv::Vec3f(0, 0, -70));
    tvec = R*tvec;

    if (verbose_)
      std::cout << "yaw pitch row: " << ypr.t()
                << " position: " << tvec.t() << std::endl;

  }

  if (verbose_) {
    cv::imshow("result", img_show);
//    video_writer << img_show;
    cv::waitKey(15);
  }

  return control_poses;
}

std::vector<cv::Vec3d> Footpath::FollowPath(const cv::Mat& img_depth,
                                            const cv::Mat& img_mask,
                                            const cv::Mat& img_color) {
  std::vector<cv::Vec3d> control_poses;

  std::vector<cv::Point2f> path_middle_lane_coords;
  path_middle_lane_coords = RowSearchingReduceMethod(img_mask);
  if (path_middle_lane_coords.size() < 10)
    return control_poses;

  std::vector<cv::Point2f> fitting_coords;
  Smooth(path_middle_lane_coords, fitting_coords);
  fitting_coords.swap(path_middle_lane_coords);

  cv::Mat img_show;
  if (verbose_) {
    if (!img_color.empty())
      img_color.copyTo(img_show, img_mask);
    else
      cv::cvtColor(img_mask, img_show, cv::COLOR_GRAY2RGB);

    int cnt = 0;
    for (const auto &it : path_middle_lane_coords) {
      cv::circle(img_show, it, 1,
                 GetColor(cnt++, 0, path_middle_lane_coords.size()));
    }
  }

  // Get the control points index
  auto control_points_index =
      GetControlPointsIndex(path_middle_lane_coords, 45);
  if (control_points_index.empty())
    return control_poses;

  // Get the control point and control pose
  control_poses.reserve(control_points_index.size());
  std::vector<cv::Point2f> undist_path_middle_lane_coords;
  cv::undistortPoints(path_middle_lane_coords, undist_path_middle_lane_coords,
                      intrinsics_, distortion_coeffs_,
                      cv::Mat(), cv::Matx33d::eye());
  for (const auto& control_point_index : control_points_index) {
    // Get the control point
    float depth = img_depth.at<uint16_t>(
        path_middle_lane_coords.at(control_point_index).x,
        path_middle_lane_coords.at(control_point_index).y) / 1000.f;
    if (depth < 0.3) continue;
    cv::Vec3f control_point =
        cv::Vec3f(undist_path_middle_lane_coords.at(control_point_index).x,
                  undist_path_middle_lane_coords.at(control_point_index).y,
                  1) * depth;

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

    if (verbose_) {
      cv::circle(img_show,
                 cv::Point(path_middle_lane_coords.at(control_point_index).x,
                           path_middle_lane_coords.at(control_point_index).y),
                 5, cv::Scalar(127, 127, 127), 2);
      std::cout << " position: " << next_pos.t() << std::endl;
    }

  }

  if (verbose_) {
    cv::imshow("result", img_show);
//    video_writer << img_show;
    cv::waitKey(0);
  }

  return control_poses;
}

std::vector<cv::Point2f> Footpath::RowSearchingReduceMethod(
    const cv::Mat& img_mask) {
  return row_searching_reduce_method(img_mask);
}

cv::Vec3f Footpath::GetIntersectPointFromLP(
    const cv::Vec3f& plane_normal_vector,
    const float plane_intercept,
    const cv::Vec3f& point) {
  cv::Vec3f line_normal_vector = cv::normalize(point);
  cv::Vec3f result =
      point - (plane_normal_vector.dot(point) + plane_intercept) /
          (plane_normal_vector.dot(line_normal_vector)) * line_normal_vector;

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
    const std::vector<cv::Point2f>& path_middle_lane, int step) {
  int number_control_point = path_middle_lane.size() / step;
  std::vector<int> result_index;
  result_index.reserve(number_control_point);
  for (int i = 0; i < number_control_point - 1; ++i)
    result_index.emplace_back(step * i + step);

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
  double r = atan2(a(0) * sin(y) - a(1) * cos(y),
                   -o(0) * sin(y) + o(1) * cos(y));
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
  double r = atan2(a(0) * sin(y) - a(1) * cos(y),
                   -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr / M_PI * 180.0;
}

cv::Matx33d Footpath::ypr2R(cv::Vec3d ypr) {
  double y = ypr(0) / 180.0 * M_PI;
  double p = ypr(1) / 180.0 * M_PI;
  double r = ypr(2) / 180.0 * M_PI;

  cv::Matx33d Rz(cos(y), -sin(y), 0,
                 sin(y), cos(y), 0,
                 0, 0, 1);

  cv::Matx33d  Ry(cos(p), 0., sin(p),
                  0., 1., 0.,
                  -sin(p), 0., cos(p));

  cv::Matx33d Rx(1., 0., 0.,
                 0., cos(r), -sin(r),
                 0., sin(r), cos(r));

  return Rz * Ry * Rx;
}

cv::Scalar Footpath::GetColor(int index, float min_range, float max_range) {
  int idx = fminf(index - min_range,
                  max_range - min_range) / (max_range - min_range) * 127.0f;
  idx = 127 - idx;

  cv::Scalar color;
  color[0] = colormapJet[idx][2] * 255.0f;
  color[1] = colormapJet[idx][1] * 255.0f;
  color[2] = colormapJet[idx][0] * 255.0f;

  return color;
}

float Footpath::colormapJet[128][3] = {
    {0.0f,0.0f,0.53125f},
    {0.0f,0.0f,0.5625f},
    {0.0f,0.0f,0.59375f},
    {0.0f,0.0f,0.625f},
    {0.0f,0.0f,0.65625f},
    {0.0f,0.0f,0.6875f},
    {0.0f,0.0f,0.71875f},
    {0.0f,0.0f,0.75f},
    {0.0f,0.0f,0.78125f},
    {0.0f,0.0f,0.8125f},
    {0.0f,0.0f,0.84375f},
    {0.0f,0.0f,0.875f},
    {0.0f,0.0f,0.90625f},
    {0.0f,0.0f,0.9375f},
    {0.0f,0.0f,0.96875f},
    {0.0f,0.0f,1.0f},
    {0.0f,0.03125f,1.0f},
    {0.0f,0.0625f,1.0f},
    {0.0f,0.09375f,1.0f},
    {0.0f,0.125f,1.0f},
    {0.0f,0.15625f,1.0f},
    {0.0f,0.1875f,1.0f},
    {0.0f,0.21875f,1.0f},
    {0.0f,0.25f,1.0f},
    {0.0f,0.28125f,1.0f},
    {0.0f,0.3125f,1.0f},
    {0.0f,0.34375f,1.0f},
    {0.0f,0.375f,1.0f},
    {0.0f,0.40625f,1.0f},
    {0.0f,0.4375f,1.0f},
    {0.0f,0.46875f,1.0f},
    {0.0f,0.5f,1.0f},
    {0.0f,0.53125f,1.0f},
    {0.0f,0.5625f,1.0f},
    {0.0f,0.59375f,1.0f},
    {0.0f,0.625f,1.0f},
    {0.0f,0.65625f,1.0f},
    {0.0f,0.6875f,1.0f},
    {0.0f,0.71875f,1.0f},
    {0.0f,0.75f,1.0f},
    {0.0f,0.78125f,1.0f},
    {0.0f,0.8125f,1.0f},
    {0.0f,0.84375f,1.0f},
    {0.0f,0.875f,1.0f},
    {0.0f,0.90625f,1.0f},
    {0.0f,0.9375f,1.0f},
    {0.0f,0.96875f,1.0f},
    {0.0f,1.0f,1.0f},
    {0.03125f,1.0f,0.96875f},
    {0.0625f,1.0f,0.9375f},
    {0.09375f,1.0f,0.90625f},
    {0.125f,1.0f,0.875f},
    {0.15625f,1.0f,0.84375f},
    {0.1875f,1.0f,0.8125f},
    {0.21875f,1.0f,0.78125f},
    {0.25f,1.0f,0.75f},
    {0.28125f,1.0f,0.71875f},
    {0.3125f,1.0f,0.6875f},
    {0.34375f,1.0f,0.65625f},
    {0.375f,1.0f,0.625f},
    {0.40625f,1.0f,0.59375f},
    {0.4375f,1.0f,0.5625f},
    {0.46875f,1.0f,0.53125f},
    {0.5f,1.0f,0.5f},
    {0.53125f,1.0f,0.46875f},
    {0.5625f,1.0f,0.4375f},
    {0.59375f,1.0f,0.40625f},
    {0.625f,1.0f,0.375f},
    {0.65625f,1.0f,0.34375f},
    {0.6875f,1.0f,0.3125f},
    {0.71875f,1.0f,0.28125f},
    {0.75f,1.0f,0.25f},
    {0.78125f,1.0f,0.21875f},
    {0.8125f,1.0f,0.1875f},
    {0.84375f,1.0f,0.15625f},
    {0.875f,1.0f,0.125f},
    {0.90625f,1.0f,0.09375f},
    {0.9375f,1.0f,0.0625f},
    {0.96875f,1.0f,0.03125f},
    {1.0f,1.0f,0.0f},
    {1.0f,0.96875f,0.0f},
    {1.0f,0.9375f,0.0f},
    {1.0f,0.90625f,0.0f},
    {1.0f,0.875f,0.0f},
    {1.0f,0.84375f,0.0f},
    {1.0f,0.8125f,0.0f},
    {1.0f,0.78125f,0.0f},
    {1.0f,0.75f,0.0f},
    {1.0f,0.71875f,0.0f},
    {1.0f,0.6875f,0.0f},
    {1.0f,0.65625f,0.0f},
    {1.0f,0.625f,0.0f},
    {1.0f,0.59375f,0.0f},
    {1.0f,0.5625f,0.0f},
    {1.0f,0.53125f,0.0f},
    {1.0f,0.5f,0.0f},
    {1.0f,0.46875f,0.0f},
    {1.0f,0.4375f,0.0f},
    {1.0f,0.40625f,0.0f},
    {1.0f,0.375f,0.0f},
    {1.0f,0.34375f,0.0f},
    {1.0f,0.3125f,0.0f},
    {1.0f,0.28125f,0.0f},
    {1.0f,0.25f,0.0f},
    {1.0f,0.21875f,0.0f},
    {1.0f,0.1875f,0.0f},
    {1.0f,0.15625f,0.0f},
    {1.0f,0.125f,0.0f},
    {1.0f,0.09375f,0.0f},
    {1.0f,0.0625f,0.0f},
    {1.0f,0.03125f,0.0f},
    {1.0f,0.0f,0.0f},
    {0.96875f,0.0f,0.0f},
    {0.9375f,0.0f,0.0f},
    {0.90625f,0.0f,0.0f},
    {0.875f,0.0f,0.0f},
    {0.84375f,0.0f,0.0f},
    {0.8125f,0.0f,0.0f},
    {0.78125f,0.0f,0.0f},
    {0.75f,0.0f,0.0f},
    {0.71875f,0.0f,0.0f},
    {0.6875f,0.0f,0.0f},
    {0.65625f,0.0f,0.0f},
    {0.625f,0.0f,0.0f},
    {0.59375f,0.0f,0.0f},
    {0.5625f,0.0f,0.0f},
    {0.53125f,0.0f,0.0f},
    {0.5f,0.0f,0.0f}
};
