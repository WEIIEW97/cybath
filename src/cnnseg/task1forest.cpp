
#include "onnxruntimeEngine.h"
#include "task1forest.h"
#include "common.h"
#include "../constants.h"
#ifndef _WIN32
#include <dirent.h>
#endif

#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

#define FORESTDEBUGMODE 1

// mask: road border: objlabel: 128->2 black, 255->1 white
//
int replaceBorderMaskLabel(Mat& mask, Mat& out_newlabelmask) {
  //
  uchar scaleForDebug = 50; // 1;

  out_newlabelmask = Mat::zeros(mask.rows, mask.cols, CV_8UC1);

  // replace
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      uchar pixel = mask.at<uchar>(i, j);
      if (pixel < 100)
        // background
        out_newlabelmask.at<uchar>(i, j) = (uchar)(0 * scaleForDebug);
      else if (pixel < 200)
        // black line
        out_newlabelmask.at<uchar>(i, j) = (uchar)(2 * scaleForDebug);
      else
        // white line
        out_newlabelmask.at<uchar>(i, j) = (uchar)(1 * scaleForDebug);
    }
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32

ForestTask::ForestTask(const wchar_t* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_border = new OnnxRuntimeEngine(border_model_path);

  this->outbuffer_border = (void*)malloc(512 * 512 * sizeof(int64_t));
};

#else
ForestTask::ForestTask(const char* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_border = new OnnxRuntimeEngine(border_model_path);

  this->outbuffer_border = (void*)malloc(512 * 512 * sizeof(int64_t));
};
#endif

ForestTask::~ForestTask() {
  free(outbuffer_border);
  delete (ortengine_border);
}

int ForestTask::processMask(cv::Mat src, cv::Mat& finalmask) {

  int res;
  float min_mask_area_ratio = 0.03;
  clock_t startTime, endTime;

  cv::Mat final_blackmask, final_whitemask, newlabel_finalmask;

  // initialize return
  finalmask = Mat::zeros(src.rows, src.cols, CV_8UC1);

  memset(this->outbuffer_border, 0, 512 * 512 * sizeof(int64_t));
  int padw, padh;

  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image is empty\n");
    return ONNXRUNTIMEENGINE_FILE_ERROR;
  }

  startTime = clock();

  // border
  res = ortengine_border->processSeg(src, Size(512, 512), padw, padh,
                                     this->outbuffer_border);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "border model processSeg error\n");
    return res;
  }

  // border: objlabel: 1 black, 2 white
  res = ortengine_border->segmentationPost(this->outbuffer_border,
                                           final_blackmask, 512, 512, padw,
                                           padh, src.rows, src.cols, 1);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "black segmentationPost error\n");
    return res;
  }

  // imwrite("./out_blackmask.jpg", final_blackmask);

  res = ortengine_border->segmentationPost(this->outbuffer_border,
                                           final_whitemask, 512, 512, padw,
                                           padh, src.rows, src.cols, 2);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "white segmentationPost error\n");
    return res;
  }

  // imwrite("./out_whitemask.jpg", final_whitemask);

  cv::addWeighted(finalmask, 1.0, final_blackmask, 0.5, 0, finalmask);
  cv::addWeighted(finalmask, 1.0, final_whitemask, 1.0, 0, finalmask);

  // replace label
  res = replaceBorderMaskLabel(finalmask, newlabel_finalmask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./newlabel_finalmask.jpg", newlabel_finalmask);

  finalmask = newlabel_finalmask;

  endTime = clock();
  // cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
  // << endl;

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32
int mainForestTask() {
  // initialize onnxruntime engine
  ForestTask* foresttask = new ForestTask(border_model_path);

  std::string filepath = "./";
  std::string outfilepath = "./";

  int imgcount = 1;
  for (int i = 0; i < imgcount; i++) {
    int res;
    Mat src, finalmask;

    // std::stringstream ss;
    // ss << std::setfill('0') << std::setw(6) << i << ".jpg";
    ////ss << i << ".jpg";
    //
    // std::string imageName = ss.str();

    std::string imageName = "1704955291.502696.jpg";

    src = imread(filepath + imageName, IMREAD_COLOR); // read image

    if (src.empty()) // check image is valid
    {
      fprintf(stderr, "ForestTask Can not load image %s\n", imageName.c_str());
      continue;
    }

    printf("now process %s\n", imageName.c_str());

    res = foresttask->processMask(src, finalmask);
    if (res != ONNXRUNTIMEENGINE_SUCCESS) {
      fprintf(stderr, "ForestTask processMask error %s\n", imageName.c_str());
      continue;
    }

    // check
    // for (int i = 0; i < finalmask.rows; i++)
    //{
    //	for (int j = 0; j < finalmask.cols; j++)
    //	{
    //		uchar pixel = finalmask.at<uchar>(i, j);
    //		if (pixel != 0 && pixel != 50 && pixel != 100 && pixel != 150 && pixel
    //!= 200 && pixel != 250) 			std::cout << i << "," << j << std::endl;
    //
    //	}
    //}

    // output to text file
    // std::ofstream file("output.txt");
    // if (!file.is_open()) {
    //	std::cerr << "Failed to open file for writing." << std::endl;
    //	return 1;
    //}
    //
    // for (int i = 0; i < finalmask.rows; ++i) {
    //	for (int j = 0; j < finalmask.cols; ++j) {
    //		file << static_cast<int>(finalmask.at<uchar>(i, j)) << " ";
    //	}
    //	file << std::endl;
    //}
    //
    // file.close();

#if FORESTDEBUGMODE
    // imageName is string
    std::string savepath = outfilepath +
                           imageName.substr(0, imageName.length() - 4) +
                           "_result.jpg";
    imwrite(savepath, finalmask);
#endif
  }

  delete (foresttask);

  return 0;
}
#else
int mainForestTask() {
  // initialize onnxruntime engine
  ForestTask* foresttask = new ForestTask(border_model_path);

  std::string filepath = "../test/";
  std::string outfilepath = "../result/";

  DIR* dp = nullptr;
  const std::string& exten = "*";
  struct dirent* dirp = nullptr;
  if ((dp = opendir(filepath.c_str())) == nullptr) {
    fprintf(stderr, "scandir error %s\n", filepath.c_str());
    return ONNXRUNTIMEENGINE_PATH_ERROR;
  }

  int i = 0;
  while ((dirp = readdir(dp)) != nullptr) {
    int res;
    Mat src, finalmask;

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
      fprintf(stderr, "ForestTask Can not load image %s\n", imageName.c_str());
      continue;
    }

    printf("now process %s\n", imageName.c_str());

    res = foresttask->processMask(src, finalmask);
    if (res != ONNXRUNTIMEENGINE_SUCCESS) {
      fprintf(stderr, "ForestTask processMask error %s\n", imageName.c_str());
      continue;
    }
#if FORESTDEBUGMODE
    // imageName is string
    std::string savepath = outfilepath +
                           imageName.substr(0, imageName.length() - 4) +
                           "_result.jpg";
    imwrite(savepath, finalmask);
#endif
  }

  delete (foresttask);

  return 0;
}

#endif
