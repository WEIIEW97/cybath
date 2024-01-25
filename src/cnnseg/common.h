#ifndef COMMON
#define COMMON

#include "onnxruntimeEngine.h"

#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

typedef struct {
  int label;
  int area;
} LabelArea;

bool labelareacmp(LabelArea x, LabelArea y);

int FilterNoisyArea(Mat& mask, float min_mask_area_ratio, Mat& outmask);

// sort x first , then y
bool xycompare(const BoundingBox& a, const BoundingBox& b);

bool xcompare(const BoundingBox& a, const BoundingBox& b);

bool ycompare(const BoundingBox& a, const BoundingBox& b);

int overlapArea(const BoundingBox& r1, const BoundingBox& r2);

int boxarea(const BoundingBox& r1);

// models
#ifdef _WIN32
static const wchar_t* border_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/"
    L"ocrnet_hr18_4xb4-80k_ade20k-512x512_border_0124/"
    L"end2end_ocrnet_border.onnx";
static const wchar_t* road_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/"
    L"ocrnet_hr18_4xb4-80k_ade20k-512x512_road_0124/end2end_ocrnet_road.onnx";
static const wchar_t* line_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/"
    L"ocrnet_hr18_4xb4-80k_ade20k-512x512_mangdaoline_0121/"
    L"end2end_ocrnet_line.onnx";
static const wchar_t* tablet_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/"
    L"yolox_s_8xb8-300e_coco_shelvetablet_0124/end2end_yolox_tablet.onnx";
static const wchar_t* coco_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/yolox_s_8xb8-300e_coco/"
    L"end2end_yolox_coco.onnx";
static const wchar_t* furnitures_model_path =
    L"D:/cybathlon_code/cnnseg/models_0124/"
    L"yolox_s_8xb8-300e_coco_furnitures_0124/end2end_yolox_furnitures.onnx";
#else
static const char* border_model_path =
    "../models_0124/ocrnet_hr18_4xb4-80k_ade20k-512x512_border_0124/"
    "end2end_ocrnet_border.onnx";
static const char* road_model_path =
    "../models_0124/ocrnet_hr18_4xb4-80k_ade20k-512x512_road_0124/"
    "end2end_ocrnet_road.onnx";
static const char* line_model_path =
    "../models_0124/ocrnet_hr18_4xb4-80k_ade20k-512x512_mangdaoline_0121/"
    "end2end_ocrnet_line.onnx";
static const char* tablet_model_path =
    "../models_0124/yolox_s_8xb8-300e_coco_shelvetablet_0124/"
    "end2end_yolox_tablet.onnx";
static const char* coco_model_path =
    "../models_0124/yolox_s_8xb8-300e_coco/end2end_yolox_coco.onnx";
static const char* furnitures_model_path =
    "../models_0124/yolox_s_8xb8-300e_coco_furnitures_0124/"
    "end2end_yolox_furnitures.onnx";
static const char* fingertip_model_path =
    "../models_0124/yolox_s_8xb8-300e_coco_fingertip_0124/"
    "end2end_yolox_fingertip.onnx";
static const std::string det_model_dir =
    "../models_0124/PP-OCR/ch_PP-OCRv4_det_infer"; // ch_PP-OCRv4_det_server_infer
static const std::string cls_model_dir =
    "../models_0124/PP-OCR/ch_ppocr_mobile_v2.0_cls_infer";
static const std::string rec_model_dir =
    "../models_0124/PP-OCR/ch_PP-OCRv3_rec_infer";
static const std::string rec_label_file =
    "../models_0124/PP-OCR/ppocr_keys_v1.txt";
#endif

#endif
