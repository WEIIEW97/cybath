// Projection factor of project to one frame depth camera
// Copyright (c) 2023, Algorithm Development Team. All rights reserved.
//
// This software was developed of Jacob.lsx

#ifndef FOOTPATH_H_
#define FOOTPATH_H_

#include <vector>
#include <opencv2/core/core.hpp>

class Footpath {
 public:
  typedef std::vector<std::pair<cv::Vec3d, cv::Vec3d>> ControlPoses;

  Footpath(const cv::Matx33d& intrinsics,
           const cv::Vec4d& distortion_coeffs,
           float angle = 70,
           bool verbose = false);
  ~Footpath();

  ControlPoses GetControlPoses(const cv::Mat& depth,
                               const cv::Mat& binary_mask,
                               const cv::Mat& color = cv::Mat());

  std::vector<cv::Vec3d> FollowPath(const cv::Mat& depth,
                                    const cv::Mat& binary_mask,
                                    const cv::Mat& color = cv::Mat());

 private:
  std::vector<cv::Point2f> RowSearchingReduceMethod(const cv::Mat& img_mask);

  void Smooth(const std::vector<cv::Point2f>& input,
              std::vector<cv::Point2f>& output);

  cv::Vec3f GetIntersectPointFromLP(const cv::Vec3f& plane_normal_vector,
                                    const float plane_intercept,
                                    const cv::Vec3f& point);

  std::vector<int> GetControlPointsIndex(
      const std::vector<cv::Point2f>& path_middle_lane, int step);

  static cv::Vec3d Rvec2ypr(const cv::Vec3d& rvec);
  static cv::Vec3d R2ypr(const cv::Matx33d& R);
  static cv::Matx33d ypr2R(cv::Vec3d ypr);

  static cv::Scalar GetColor(int index, float min_range, float max_range);

  cv::Matx33d intrinsics_;
  cv::Vec4d distortion_coeffs_;
  float angle_;
  bool verbose_;
  cv::Matx33d R_cam_body_;

  static float colormapJet[128][3];
};


#endif // FOOTPATH_H_
