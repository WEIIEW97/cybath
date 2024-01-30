/*
 * Copyright (c) 2022-2024, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "serial.h"
#include "constants.h"
#include "centorid/detect_center_line.h"
#include "depthnavi/depth_navigation.h"
#include "startline/detect_start_line.h"

#define LABEL_SCALAR                              50
#define V_GAP_CENTER_AND_CONTROL_POINT_DIFF_PIXEL 100
#define GAP_CENTER_AND_CONTROL_POINT_DIFF_PIXEL   50

#if 0
ortPathSegGPU* initialize_gpu(const std::string& road_onnx_model_path,
                              const std::string& line_onnx_model_path) {
  auto* ort_pathseg_gpu =
      new ortPathSegGPU(road_onnx_model_path, line_onnx_model_path);
  return ort_pathseg_gpu;
}
#endif

#if 0
cv::Mat onnx_path_seg(const cv::Mat& frame, ortPathSegGPU* stream) {
  cv::Mat onnx_seg;
  auto res = stream->processMask(frame, onnx_seg);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    std::cerr << "processing segment error."
              << "\n";
  }
  return onnx_seg;
}
#endif

void get_labeled_masks_from_onnx(
    const cv::Mat& onnx_seg_result,
    std::shared_ptr<MultiLabelMaskSet>& multi_label_masks) {

  // split multi-labeled tasks into different masks;
  multi_label_masks->global_start_end_lane =
      (onnx_seg_result == 1 * LABEL_SCALAR);
  multi_label_masks->border_lane = (onnx_seg_result == 2 * LABEL_SCALAR);
  multi_label_masks->shape_v_lane = (onnx_seg_result == 3 * LABEL_SCALAR);
  multi_label_masks->gap_lane = (onnx_seg_result == 4 * LABEL_SCALAR);
  multi_label_masks->road_lane = (onnx_seg_result == 5 * LABEL_SCALAR);

  multi_label_masks->global_start_end_lane *= 255;
  multi_label_masks->border_lane *= 255;
  multi_label_masks->shape_v_lane *= 255;
  multi_label_masks->gap_lane *= 255;
  multi_label_masks->road_lane *= 255;
}

Case1Package
serial_start_line_detect(std::shared_ptr<MultiLabelMaskSet>& label_masks) {
  auto vertices = get_rectangle_vertices(label_masks->global_start_end_lane);
  Case1Package start_line_signal;
  if (vertices.size() == 1) {
    return start_line_signal;
  }
#ifdef DEBUG
  std::cout << "vertices is: "
            << "\n";
  for (auto& p : vertices) {
    std::cout << p << " ";
  }
  std::cout << "\n";
#endif
  cv::Point2f c, unit_v;
  PositionFlag flag;
  fit_rectangle(vertices, c, unit_v, flag);
  cv::Point2f v2 = {-1, 0};
#ifdef DEBUG
  std::cout << "center is " << c << std::endl;
#endif
  auto theta = calculate_theta(unit_v, v2);
  auto angle = rad2deg(theta);
#ifdef DEBUG
  std::cout << ">>> unit vector is:  " << unit_v << "\n";
  std::cout << ">>> theta is: " << theta << "\n";
  std::cout << ">>> angle is: " << angle << "\n";
  std::cout << ">>> direction flag is: " << flag << "\n";
#endif

  start_line_signal.angle = angle;
  flag = (angle < 1.0f) ? PositionFlag::align : flag;
  start_line_signal.sign = flag;
  return start_line_signal;
}

Case2Package
serial_center_line_detect(std::shared_ptr<MultiLabelMaskSet>& label_masks,
                          Footpath& footpath, const cv::Mat& correspond_depth,
                          float indicate_thr1, float indicate_thr2,
                          float indicate_thr3) {
  Case2Package msg;
  auto mask = label_masks->road_lane;
  auto v_mask = label_masks->shape_v_lane;
  auto gap_mask = label_masks->gap_lane;

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

  auto control_poses =
      footpath.FollowPath(correspond_depth, mask, label_masks->gap_lane);

  auto nearest_point = footpath.nearest_control_point_coord_;
#ifdef DEBUG
  std::cout << ">>>nearest point: " << nearest_point << std::endl;
#endif

  auto gap_centorid_point = find_gap_centorid(v_mask);
#ifdef DEBUG
  std::cout << ">>>v gap center point: " << gap_centorid_point << std::endl;
#endif

  // ignore the offset of the direction of x, just target on y
  // if gap centroid point

  /// for step up/down functionality
  if (nearest_point.y - gap_centorid_point.y <
          V_GAP_CENTER_AND_CONTROL_POINT_DIFF_PIXEL &&
      nearest_point.y - gap_centorid_point.y >= 0) {
    auto l2_dist = cv::sqrt(control_poses[0][1] * control_poses[0][1] +
                            control_poses[0][2] * control_poses[0][2]);
#ifdef DEBUG
    std::cout << "l2 dist: " << l2_dist << "\n";
#endif
    //
    if (footpath.step_up_sign_ && !footpath.is_v_mask_exists_)
      footpath.is_have_seen_step_up_sign_ = true;

    if (!footpath.is_have_seen_step_up_sign_) {
      if (l2_dist < indicate_thr1) {
        msg.step_up_sign = true;
        footpath.step_up_sign_ = true;
      }
    } else {
      if (l2_dist < indicate_thr2) {
        msg.step_down_sign = true;
        footpath.step_down_sign_ = true;
      }
    }
  }

  /// for gap warning functionality
  cv::Point bottom_mask_center_coord =
      footpath.CalculateBottomGapMaskCenterCoord(gap_mask);
#ifdef DEBUG
  std::cout << ">>>bottom cautioius gap center point: "
            << bottom_mask_center_coord << std::endl;
#endif
  if (nearest_point.y - bottom_mask_center_coord.y <
          GAP_CENTER_AND_CONTROL_POINT_DIFF_PIXEL &&
      nearest_point.y - bottom_mask_center_coord.y >= 0) {
    auto l2_dist = cv::sqrt(control_poses[0][1] * control_poses[0][1] +
                            control_poses[0][2] * control_poses[0][2]);
#ifdef DEBUG
    std::cout << "l2 dist: " << l2_dist << "\n";
#endif
    if (l2_dist < indicate_thr3) {
      msg.mind_gap = true;
    }
  }
  msg.data = control_poses;
  return msg;
}

Case3Package serial_navigate_by_depth_and_box_3(
    bool has_cabinet, std::vector<int>& cabin_pos, bool has_tab,
    std::vector<int>& tab_pos, const cv::Mat& aligned_depth, float thr) {
  Case3Package msg{};

  int TOLERANCE_PIXEL_THR = 20;

  int half_w = aligned_depth.cols / 2;

  float global_mean_dpeth;
  float mean_cabinet_depth = 0.f;
  float mean_tab_depth = 0.f;

  if (has_cabinet) {
    int cabin_trunc_x = (cabin_pos[2] - cabin_pos[0]) / 4;
    int cabin_trunc_y = (cabin_pos[3] - cabin_pos[1]) / 4;
    cv::Point upper_left = {cabin_pos[0] + cabin_trunc_x,
                            cabin_pos[1] + cabin_trunc_y};
    cv::Point bottom_right = {cabin_pos[2] - cabin_trunc_x,
                              cabin_pos[3] - cabin_trunc_y};
    cv::Rect roi(upper_left, bottom_right);
    mean_cabinet_depth = cv::mean(aligned_depth(roi))[0];

    int tab_center_x = 0;

    if (has_tab) {
      int tab_trunc_x = (tab_pos[2] - tab_pos[0]) / 4;
      int tab_trunc_y = (tab_pos[3] - tab_pos[1]) / 4;
      cv::Point upper_left = {tab_pos[0] + tab_trunc_x, tab_pos[1] + tab_trunc_y};
      cv::Point bottom_right = {tab_pos[2] - tab_trunc_x,
                                tab_pos[3] - tab_trunc_y};
      cv::Rect roi(upper_left, bottom_right);
      mean_cabinet_depth = cv::mean(aligned_depth(roi))[0];
      tab_center_x = (tab_pos[0] + tab_pos[2]) / 2;
    }


    global_mean_dpeth = mean_tab_depth > 0.f ? (mean_cabinet_depth + mean_tab_depth) / 2 : mean_cabinet_depth;
    msg.depth = global_mean_dpeth;
    /// very brutal and simple way to tell whether to turn left or right
    auto cabin_center_x = (cabin_pos[0] + cabin_pos[2]) / 2;
    auto bb_center_x = tab_center_x > 0 ? (cabin_center_x + tab_center_x) / 2 : cabin_center_x;

    very_sloppy_left_right_turn_indicator(half_w, bb_center_x,
                                          TOLERANCE_PIXEL_THR, msg.sign);

    if (global_mean_dpeth <= thr) {
      msg.sign = PositionFlag::stop;
    }
  }
  return msg;
}

Case3Package serial_navigate_by_depth_and_box_4(bool has_cabinet,
                                                std::vector<int>& position,
                                                const cv::Mat& rgb,
                                                const cv::Mat& aligned_depth,
                                                float dist_1, float dist_2) {
  Case3Package msg{};
  float sigma = 30;
  uint8_t black_thr = 50;
  int TOLERANCE_PIXEL_THR = 20;
  int three_quarters_w = aligned_depth.cols / 4 * 3;

  float distance = 0.f;

  if (has_cabinet) {
    cv::Point upper_left = {position[0], position[1]};
    cv::Point bottom_right = {position[2], position[3]};
    cv::Rect roi(upper_left, bottom_right);
    auto rgb_block = rgb(roi);
    auto aligned_depth_block = aligned_depth(roi);

    cv::Scalar lb(0, 0, 0);
    cv::Scalar ub(black_thr, black_thr, black_thr);
    cv::Mat black_mask;
    cv::inRange(rgb_block, lb, ub, black_mask);

    /// take the black area mask;
    distance = cv::mean(aligned_depth_block, black_mask)[0];
    if (distance - dist_1 <= sigma)
      msg.sign = PositionFlag::stop_at_one;

    if (distance - dist_2 <= sigma)
      msg.sign = PositionFlag::stop_at_two;

    auto position_x = (position[0] + position[2]) / 2;
    very_sloppy_left_right_turn_indicator(three_quarters_w, position_x,
                                          TOLERANCE_PIXEL_THR, msg.sign);
    msg.depth = distance;
  }
  return msg;
}

bool whether_to_begin_construction(const Case1Package& signal) {
  return (signal.angle < 1.0f);
}

#if 0
void delete_gpu(ortPathSegGPU* GPU) { delete (GPU); }
#endif