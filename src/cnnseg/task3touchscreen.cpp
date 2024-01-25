// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

#include <vector>
#include <algorithm>

#include "task3touchscreen.h"
#include "common.h"

using namespace cv;
using namespace std;

#define TORCHSCREENBUGMODE 1

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

/**********************************************************************

void InitAndInfer(const std::string &det_model_dir,
                  const std::string &cls_model_dir,
                  const std::string &rec_model_dir,
                  const std::string &rec_label_file,
                  const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  // The cls and rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int cls_batch_size = 1;
  int rec_batch_size = 6;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                              {cls_batch_size, 3, 48, 1024});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});

  // Users could save TRT cache file to disk as follow.
  // det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  // cls_option.SetTrtCacheFile(cls_model_dir + sep + "cls_trt_cache.trt");
  // rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");

  auto det_model = fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  auto cls_model = fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  auto rec_model = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);

  assert(det_model.Initialized());
  assert(cls_model.Initialized());
  assert(rec_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model.GetPreprocessor().SetMaxSideLen(960);
  det_model.GetPostprocessor().SetDetDBThresh(0.3);
  det_model.GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model.GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model.GetPostprocessor().SetDetDBScoreMode("slow");
  det_model.GetPostprocessor().SetUseDilation(0);
  cls_model.GetPostprocessor().SetClsThresh(0.9);

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  // auto ppocr_v3 = fastdeploy::pipeline::PPOCRv3(&det_model, &rec_model);
  auto ppocr_v3 =
      fastdeploy::pipeline::PPOCRv3(&det_model, &cls_model, &rec_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size
  // of the cls and rec models will be the same as the number of boxes detected
  // by the det model.
  ppocr_v3.SetClsBatchSize(cls_batch_size);
  ppocr_v3.SetRecBatchSize(rec_batch_size);

  if (!ppocr_v3.Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  auto im_bak = im.clone();

  fastdeploy::vision::OCRResult result;
  if (!ppocr_v3.Predict(&im, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << result.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisOcr(im_bak, result);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}


int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cout << "Usage: infer_demo path/to/det_model path/to/cls_model "
                 "path/to/rec_model path/to/rec_label_file path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./ch_PP-OCRv3_det_infer "
                 "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer "
                 "./ppocr_keys_v1.txt ./12.jpg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, e.g. 0: run with paddle "
                 "inference on cpu;"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[6]);

  if (flag == 0) {
    option.UseCpu();
    option.UsePaddleBackend(); // Paddle Inference
  } else if (flag == 1) {
    option.UseCpu();
    option.UseOpenVINOBackend(); // OpenVINO
  } else if (flag == 2) {
    option.UseCpu();
    option.UseOrtBackend(); // ONNX Runtime
  } else if (flag == 3) {
    option.UseCpu();
    option.UseLiteBackend(); // Paddle Lite
  } else if (flag == 4) {
    option.UseGpu();
    option.UsePaddleBackend(); // Paddle Inference
  } else if (flag == 5) {
    option.UseGpu();
    option.UsePaddleInferBackend();
    option.paddle_infer_option.collect_trt_shape = true;
    option.paddle_infer_option.enable_trt = true; // Paddle-TensorRT
  } else if (flag == 6) {
    option.UseGpu();
    option.UseOrtBackend(); // ONNX Runtime
  } else if (flag == 7) {
    option.UseGpu();
    option.UseTrtBackend(); // TensorRT
  }

  std::string det_model_dir = argv[1];
  std::string cls_model_dir = argv[2];
  std::string rec_model_dir = argv[3];
  std::string rec_label_file = argv[4];
  std::string test_image = argv[5];
  InitAndInfer(det_model_dir, cls_model_dir, rec_model_dir, rec_label_file,
               test_image, option);
  return 0;
}
**********************************************************************/

typedef struct {
  std::array<int, 8> box;
  std::string label; // int64
} TextBox;

typedef struct {
  int index;
  std::string matchedStr; // int64
} StringMatchRes;

bool TextBox_ycompare(const TextBox& a, const TextBox& b) {
  return (a.box[1] + a.box[3] + a.box[5] + a.box[7]) * 1.0 / 4 <
         (b.box[1] + b.box[3] + b.box[5] + b.box[7]) * 1.0 / 4;
}

// edit distance of two string
int levenshteinDistance(const std::string& s1, const std::string& s2) {
  int m = s1.size();
  int n = s2.size();
  std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

  for (int i = 0; i <= m; ++i) {
    dp[i][0] = i;
  }

  for (int j = 0; j <= n; ++j) {
    dp[0][j] = j;
  }

  for (int j = 1; j <= n; ++j) {
    for (int i = 1; i <= m; ++i) {
      if (s1[i - 1] == s2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = std::min(
            {dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1});
      }
    }
  }

  return dp[m][n];
}

// string vector find the minimal editdistance string
StringMatchRes findClosestString(const std::vector<std::string>& strings,
                                 const std::string& target) {
  int minDistance = std::numeric_limits<int>::max();
  StringMatchRes closestString;
  closestString.index = -1;

  for (int i = 0; i < strings.size(); i++) {
    string s = strings[i];
    int distance = levenshteinDistance(s, target);
    if (distance < minDistance) {
      minDistance = distance;

      closestString.index = i;
      closestString.matchedStr = s;
    }
  }

  return closestString;
}

// remove non english alphabet and to lower
std::string removeNonAlphabeticAndToLower(std::string str) {
  str.erase(
      std::remove_if(
          str.begin(), str.end(),
          [](char c) { return !std::isalpha(static_cast<unsigned char>(c)); }),
      str.end());

  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  return str;
}

TouchscreenTask::TouchscreenTask(const char* fingertip_model_path,
                                 const std::string det_model_dir,
                                 const std::string cls_model_dir,
                                 const std::string rec_model_dir,
                                 const std::string rec_label_file) {
  // initialize onnxruntime engine
  ortengine_fingertip = new OnnxRuntimeEngine(fingertip_model_path);

  // ocr models
  det_model = NULL;
  cls_model = NULL;
  rec_model = NULL;
  ppocr_v3 = NULL;

  fastdeploy::RuntimeOption option;
  option.UseGpu();
  option.UsePaddleBackend(); // Paddle Inference

  auto det_model_file = det_model_dir + sep + "inference.pdmodel";
  auto det_params_file = det_model_dir + sep + "inference.pdiparams";

  auto cls_model_file = cls_model_dir + sep + "inference.pdmodel";
  auto cls_params_file = cls_model_dir + sep + "inference.pdiparams";

  auto rec_model_file = rec_model_dir + sep + "inference.pdmodel";
  auto rec_params_file = rec_model_dir + sep + "inference.pdiparams";

  auto det_option = option;
  auto cls_option = option;
  auto rec_option = option;

  // The cls and rec model can inference a batch of images now.
  // User could initialize the inference batch size and set them after create
  // PP-OCR model.
  int cls_batch_size = 1;
  int rec_batch_size = 6;

  // If use TRT backend, the dynamic shape will be set as follow.
  // We recommend that users set the length and height of the detection model to
  // a multiple of 32.
  // We also recommend that users set the Trt input shape as follow.
  det_option.SetTrtInputShape("x", {1, 3, 64, 64}, {1, 3, 640, 640},
                              {1, 3, 960, 960});
  cls_option.SetTrtInputShape("x", {1, 3, 48, 10}, {cls_batch_size, 3, 48, 320},
                              {cls_batch_size, 3, 48, 1024});
  rec_option.SetTrtInputShape("x", {1, 3, 48, 10}, {rec_batch_size, 3, 48, 320},
                              {rec_batch_size, 3, 48, 2304});

  // Users could save TRT cache file to disk as follow.
  // det_option.SetTrtCacheFile(det_model_dir + sep + "det_trt_cache.trt");
  // cls_option.SetTrtCacheFile(cls_model_dir + sep + "cls_trt_cache.trt");
  // rec_option.SetTrtCacheFile(rec_model_dir + sep + "rec_trt_cache.trt");

  det_model = new fastdeploy::vision::ocr::DBDetector(
      det_model_file, det_params_file, det_option);
  if (!det_model->Initialized()) {
    std::cerr << "Failed to initialize det_model." << std::endl;
    return;
  }
  cls_model = new fastdeploy::vision::ocr::Classifier(
      cls_model_file, cls_params_file, cls_option);
  if (!cls_model->Initialized()) {
    std::cerr << "Failed to initialize det_model." << std::endl;
    return;
  }
  rec_model = new fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, rec_label_file, rec_option);
  if (!rec_model->Initialized()) {
    std::cerr << "Failed to initialize rec_model." << std::endl;
    return;
  }

  // assert(det_model.Initialized());
  // assert(cls_model.Initialized());
  // assert(rec_model.Initialized());

  // Parameters settings for pre and post processing of Det/Cls/Rec Models.
  // All parameters are set to default values.
  det_model->GetPreprocessor().SetMaxSideLen(960);
  det_model->GetPostprocessor().SetDetDBThresh(0.3);
  det_model->GetPostprocessor().SetDetDBBoxThresh(0.6);
  det_model->GetPostprocessor().SetDetDBUnclipRatio(1.5);
  det_model->GetPostprocessor().SetDetDBScoreMode("slow");
  det_model->GetPostprocessor().SetUseDilation(0);
  cls_model->GetPostprocessor().SetClsThresh(0.9);

  // The classification model is optional, so the PP-OCR can also be connected
  // in series as follows
  ppocr_v3 = new fastdeploy::pipeline::PPOCRv3(det_model, rec_model);
  // ppocr_v3 =
  //     fastdeploy::pipeline::PPOCRv3(det_model, cls_model, rec_model);

  // Set inference batch size for cls model and rec model, the value could be -1
  // and 1 to positive infinity.
  // When inference batch size is set to -1, it means that the inference batch
  // size
  // of the cls and rec models will be the same as the number of boxes detected
  // by the det model.
  ppocr_v3->SetClsBatchSize(cls_batch_size);
  ppocr_v3->SetRecBatchSize(rec_batch_size);

  if (!ppocr_v3->Initialized()) {
    std::cerr << "Failed to initialize PP-OCR." << std::endl;
    return;
  }
}

int TouchscreenTask::doOCR(cv::Mat& src, cv::Mat& vis_im, std::string& orderstr,
                           std::string& targetstr,
                           std::array<int, 8>& targetbox, bool& findFingertip,
                           BoundingBox& fingertip) {
  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image is empty\n");
    return PADLEOCRENGINE_FILE_ERROR;
  }

  int res;
  fastdeploy::vision::OCRResult result;

  clock_t startTime, endTime;

  std::vector<BoundingBox> fingertip_outboxes;
  findFingertip = false;

  startTime = clock();

  if (!ppocr_v3->Predict(&src, &result)) {
    std::cerr << "Failed to predict." << std::endl;
    return PADLEOCRENGINE_PREDICT_ERROR;
  }

  // filter result
  std::vector<TextBox> postboxes;

  int numboxes = result.boxes.size();
  for (int textboxid = 0; textboxid < numboxes; textboxid++) {
    if (result.rec_scores[textboxid] <= 0.5)
      continue;

    TextBox textbox;

    textbox.box = result.boxes[textboxid];
    textbox.label = result.text[textboxid];

    postboxes.push_back(textbox);
  }

  if (postboxes.size() < 1) {
    std::cerr << "No Available TextBox." << std::endl;
    return PADLEOCRENGINE_NO_VALID_TEXTBOX_ERROR;
  }

  // sort by y
  std::sort(postboxes.begin(), postboxes.end(), TextBox_ycompare);

  // find "order"
  int orderstrpos = -1;
  for (size_t i = 0; i < postboxes.size(); ++i) {
    std::string label = removeNonAlphabeticAndToLower(postboxes[i].label);
    if (label.find("order") != std::string::npos) {
      orderstrpos = i;
      break;
    }
  }

  if (orderstrpos == -1 || orderstrpos == postboxes.size() - 1)
    return PADLEOCRENGINE_ORDER_NOT_FOUND_ERROR;

  // find item on "order"'s right
  std::string orderstring = postboxes[orderstrpos + 1].label;
  orderstr = orderstring;

  orderstring = removeNonAlphabeticAndToLower(orderstring);

  // find the minimal edit distance string box
  std::vector<std::string> menunames;
  std::vector<std::array<int, 8>> menuboxes;
  for (size_t i = 0; i < postboxes.size(); ++i) {
    if (i == orderstrpos || i == orderstrpos + 1)
      continue;
    menunames.push_back(postboxes[i].label);
    menuboxes.push_back(postboxes[i].box);
  }

  StringMatchRes closestString = findClosestString(menunames, orderstring);

  if (closestString.index == -1)
    return PADLEOCRENGINE_ORDER_NOT_IN_MENU_ERROR;

  targetbox = menuboxes[closestString.index];
  targetstr = menunames[closestString.index];

#if DEBUG_ENABLE
  // std::cout << result.Str() << std::endl;
#endif

#if TORCHSCREENBUGMODE
  cv::Mat im_bak = src.clone();
  vis_im = fastdeploy::vision::VisOcr(im_bak, result);
#else
  vis_im = src.clone();
#endif

  // cv::imwrite("vis_result.jpg", vis_im);
  // std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;

  // call finger index model
  // only get fingertip label
  res = ortengine_fingertip->processDet(src, Size(640, 640), 0, 0.5,
                                        fingertip_outboxes);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processDet error\n");
    return res;
  }

  if (fingertip_outboxes.size() < 1) {
    findFingertip = false;
    fingertip.label = 0;
    fingertip.score = 0;
    fingertip.x1 = 0;
    fingertip.y1 = 0;
    fingertip.x2 = 0;
    fingertip.y2 = 0;
  } else {
    findFingertip = true;
    fingertip = fingertip_outboxes[0];
  }

  endTime = clock();
  // cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
  // << endl;

  return PADLEOCRENGINE_SUCCESS;
}

TouchscreenTask::~TouchscreenTask() {
  if (ppocr_v3)
    delete (ppocr_v3);
  if (rec_model)
    delete (rec_model);
  if (cls_model)
    delete (cls_model);
  if (det_model)
    delete (det_model);

  if (ortengine_fingertip)
    delete (ortengine_fingertip);
}

int mainTouchscreenTask() {
  // initialize engine
  TouchscreenTask* pdl_menuocr_gpu =
      new TouchscreenTask(fingertip_model_path, det_model_dir, cls_model_dir,
                          rec_model_dir, rec_label_file);

  std::string filepath = "../test/";
  std::string outfilepath = "../result/";

  DIR* dp = nullptr;
  const std::string& exten = "*";
  struct dirent* dirp = nullptr;
  if ((dp = opendir(filepath.c_str())) == nullptr) {
    fprintf(stderr, "scandir error %s\n", filepath.c_str());
    return PADLEOCRENGINE_PATH_ERROR;
  }

  int i = 0;
  while ((dirp = readdir(dp)) != nullptr) {
    int res;
    cv::Mat src;
    cv::Mat vis_im;
    std::array<int, 8> targetbox;
    std::string targetstr;
    std::string orderstr;
    BoundingBox fingertip;
    bool findFingertip;

    std::string imageName;

    if (dirp->d_type == DT_REG) {
      if (exten.compare("*") == 0)
        imageName = dirp->d_name;
      else if (std::string(dirp->d_name).find(exten) != std::string::npos)
        imageName = dirp->d_name;
    } else
      continue;

    src = imread(filepath + imageName, IMREAD_COLOR); // read image

    if (src.empty()) // check image is valid
    {
      fprintf(stderr, "Can not load image %s\n", imageName.c_str());
      continue;
    }

    printf("now process %s\n", imageName.c_str());

    res = pdl_menuocr_gpu->doOCR(src, vis_im, orderstr, targetstr, targetbox,
                                 findFingertip, fingertip);
    if (res != PADLEOCRENGINE_SUCCESS) {
      fprintf(stderr, "process ocr error %s\n", imageName.c_str());
      continue;
    }

#if TORCHSCREENBUGMODE
    // Print the elements of the array using a range-based for loop
    for (const auto& pos : targetbox) {
      std::cout << pos << " ";
    }
    std::cout << orderstr << "    " << targetstr << std::endl;

    if (findFingertip)
      std::cout << fingertip.x1 << " " << fingertip.y1 << " " << fingertip.x2
                << " " << fingertip.y2 << std::endl;

    // draw target textbox in vis_im
    std::vector<cv::Point> corners = {cv::Point(targetbox[0], targetbox[1]),
                                      cv::Point(targetbox[2], targetbox[3]),
                                      cv::Point(targetbox[4], targetbox[5]),
                                      cv::Point(targetbox[6], targetbox[7])};

    for (size_t i = 0; i < corners.size(); ++i) {
      cv::line(vis_im, corners[i], corners[(i + 1) % corners.size()],
               cv::Scalar(255, 0, 0), 2);
    }

    if (findFingertip) {
      // draw fingertip
      int xmin = std::min(fingertip.x1, fingertip.x2);
      int ymin = std::min(fingertip.y1, fingertip.y2);
      int xmax = std::max(fingertip.x1, fingertip.x2);
      int ymax = std::max(fingertip.y1, fingertip.y2);

      std::vector<cv::Point> corners1 = {
          cv::Point(xmin, ymin), cv::Point(xmax, ymin), cv::Point(xmax, ymax),
          cv::Point(xmin, ymax)};

      for (size_t i = 0; i < corners1.size(); ++i) {
        cv::line(vis_im, corners1[i], corners1[(i + 1) % corners1.size()],
                 cv::Scalar(0, 0, 255), 2);
      }
    }

    // imageName is string
    std::string savepath = outfilepath +
                           imageName.substr(0, imageName.length() - 4) +
                           "_result.jpg";
    imwrite(savepath, vis_im);
#endif
  }

  delete (pdl_menuocr_gpu);

  return 0;
}