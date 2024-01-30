/*
 * Copyright (c) 2022-2025, William Wei. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYBATHLON_CONSTANTS_H
#define CYBATHLON_CONSTANTS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <utility>
#include <iostream>
#include <unordered_map>

void print_board(const std::unordered_map<int, std::pair<int, int>>& board);
std::unordered_map<int, std::pair<int, int>> case1_cheat_board();

static const std::string border_model_path =
    std::string(PROJECT_PATH) +
    "/models/ocrnet_hr18_4xb4-80k_ade20k-512x512_border_0125/"
    "end2end_ocrnet_border.onnx";
static const std::string road_model_path =
    std::string(PROJECT_PATH) +
    "/models/ocrnet_hr18_4xb4-80k_ade20k-512x512_road_0125/"
    "end2end_ocrnet_road.onnx";
static const std::string line_model_path =
    std::string(PROJECT_PATH) +
    "/models/ocrnet_hr18_4xb4-80k_ade20k-512x512_mangdaoline_0125/"
    "end2end_ocrnet_line.onnx";
static const std::string tablet_model_path =
    std::string(PROJECT_PATH) +
    "/models/yolox_s_8xb8-300e_coco_shelvetablet_0125/end2end_yolox_tablet.onnx";
static const std::string coco_model_path =
    std::string(PROJECT_PATH) +
    "/models/yolox_s_8xb8-300e_coco/end2end_yolox_coco.onnx";
static const std::string furnitures_model_path =
    std::string(PROJECT_PATH) +
    "/models/yolox_s_8xb8-300e_coco_furnitures_0125/"
    "end2end_yolox_furnitures.onnx";
static const std::string fingertip_model_path =
    std::string(PROJECT_PATH) + "/models/yolox_s_8xb8-300e_coco_fingertip_0125/"
                                "end2end_yolox_fingertip.onnx";
static const std::string det_model_dir =
    std::string(PROJECT_PATH) +
    "/models/PP-OCR/ch_PP-OCRv4_det_infer"; // ch_PP-OCRv4_det_server_infer
static const std::string cls_model_dir =
    std::string(PROJECT_PATH) + "/models/PP-OCR/ch_ppocr_mobile_v2.0_cls_infer";
static const std::string rec_model_dir =
    std::string(PROJECT_PATH) + "/models/PP-OCR/ch_PP-OCRv3_rec_infer";
static const std::string rec_label_file =
    std::string(PROJECT_PATH) + "/models/PP-OCR/ppocr_keys_v1.txt";

#endif // CYBATHLON_CONSTANTS_H
