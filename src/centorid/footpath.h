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
  cv::Point2f nearest_control_point_coord_;
  bool is_v_mask_exists_ = false;
  bool is_have_seen_step_up_sign_ = false;
  bool step_up_sign_ = false;
  bool step_down_sign_ = false;

public:
  typedef std::vector<std::pair<cv::Vec3d, cv::Vec3d>> ControlPoses;

  Footpath(const cv::Matx33d& intrinsics, const cv::Vec4d& distortion_coeffs,
           float angle = 70);
  ~Footpath();

  std::vector<cv::Vec3d> FollowPath(const cv::Mat& path_depth,
                                    const cv::Mat& path_mask,
                                    const cv::Mat& gap_mask);

  cv::Point CalculateBottomGapMaskCenterCoord(const cv::Mat& gap_mask);

private:
  std::vector<cv::Point2f> RowSearchingReduceMethod(const cv::Mat& img_mask);

  void Smooth(const std::vector<cv::Point2f>& input,
              std::vector<cv::Point2f>& output);

  std::vector<int>
  GetControlPointsIndex(const std::vector<cv::Point2f>& path_middle_lane,
                        const cv::Mat& gap_mask, int step = 45);

  static cv::Vec3d Rvec2ypr(const cv::Vec3d& rvec);
  static cv::Vec3d R2ypr(const cv::Matx33d& R);
  static cv::Matx33d ypr2R(cv::Vec3d ypr);

  cv::Matx33d intrinsics_;
  cv::Vec4d distortion_coeffs_;
  float angle_;
  cv::Matx33d R_cam_body_;
};

#endif // FOOTPATH_H_
