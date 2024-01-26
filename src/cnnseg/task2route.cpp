#include "onnxruntimeEngine.h"
#include "task2route.h"
#include "../constants.h"
#ifndef _WIN32
#include <dirent.h>
#endif

#include <cstdio>
#include <iostream>
#include "common.h"

#define ROUTEDEBUGMODE 1

using namespace cv;
using namespace std;

// out_linemask
int labelLineType(Mat& out_linemask, Mat& out_linetypemask) {
  out_linetypemask = Mat::zeros(out_linemask.rows, out_linemask.cols, CV_8UC1);

  // get contours
  vector<vector<Point>> contours;
  findContours(out_linemask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  // iterate each contour
  for (int i = 0; i < contours.size(); i++) {
    // find convelHull
    // vector<int> hull;
    // convexHull(contours[i], hull, false, false);
    cv::RotatedRect mar = minAreaRect(Mat(contours[i]));

    // minAreaRect area
    double mar_area = mar.size.area();

    // convexHull area
    vector<Point> points = contours[i];

    // double convex_area = 0;
    // for (int i = 0; i < hull.size(); i++)
    //{
    //	int j = (i + 1) % hull.size();
    //	convex_area += points[hull[i]].x * points[hull[j]].y - points[hull[j]].x
    //* points[hull[i]].y;
    // }

    // contour area
    double cont_area = contourArea(points);

    // check hull points
    if (mar_area / cont_area < 5.0) {
      // line
      drawContours(out_linetypemask, contours, i, 128, -1);
    } else {
      // v
      drawContours(out_linetypemask, contours, i, 255, -1);
    }
  }

  // imwrite("out_linetypemask.jpg", out_linetypemask);

  return ONNXRUNTIMEENGINE_SUCCESS;
}

// mask: road border: objlabel: 128->5 road, 25->2 black, 75->1 white, 192->4
// gapline , 255->3 startstop v line
//
int replaceRoadMaskLabel(Mat& mask, Mat& out_newlabelmask) {
  //
  uchar scaleForDebug = 50; // 1;

  out_newlabelmask = Mat::zeros(mask.rows, mask.cols, CV_8UC1);

  // replace
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      uchar pixel = mask.at<uchar>(i, j);
      if (pixel < 10)
        // background
        out_newlabelmask.at<uchar>(i, j) = (uchar)(0 * scaleForDebug);
      else if (pixel < 50)
        // black line
        out_newlabelmask.at<uchar>(i, j) = (uchar)(2 * scaleForDebug);
      else if (pixel < 100)
        // white line
        out_newlabelmask.at<uchar>(i, j) = (uchar)(1 * scaleForDebug);
      else if (pixel < 150)
        // road
        out_newlabelmask.at<uchar>(i, j) = (uchar)(5 * scaleForDebug);
      else if (pixel < 220)
        // gapline
        out_newlabelmask.at<uchar>(i, j) = (uchar)(4 * scaleForDebug);
      else
        // startstopline
        out_newlabelmask.at<uchar>(i, j) = (uchar)(3 * scaleForDebug);
    }
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int GetShoePos(std::vector<BoundingBox>& shoe_outboxes, BoundingBox& shoe) {
  bool findobj = false;
  for (auto& box : shoe_outboxes) {
    if (box.label == 0) // shelve
    {
      shoe = box;
      findobj = true;
      break;
    }
  }

  if (findobj == false) {
    shoe.label = 0;
    shoe.score = 0;
    shoe.x1 = 0;
    shoe.y1 = 0;
    shoe.x2 = 0;
    shoe.y2 = 0;
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32

RouteTask::RouteTask(const wchar_t* road_model_path,
                     const wchar_t* border_model_path,
                     const wchar_t* line_model_path) {

#if 0
	const wchar_t* shoe_model_path = L"C:/Users/Administrator/Desktop/end2end_yolox_shoe.onnx";

	// initialize onnxruntime engine
	this->ortengine_shoe = new OnnxRuntimeEngine(shoe_model_path);
#endif

  // initialize onnxruntime engine
  this->ortengine_road = new OnnxRuntimeEngine(road_model_path);
  this->ortengine_border = new OnnxRuntimeEngine(border_model_path);
  this->ortengine_line = new OnnxRuntimeEngine(line_model_path);

  this->outbuffer_road = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_border = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_line = (void*)malloc(512 * 512 * sizeof(int64_t));
};

#else
RouteTask::RouteTask(const char* road_model_path, const char* border_model_path,
                     const char* line_model_path) {
#if 0
	const char* shoe_model_path = "/algdata01/huan.wang/samlabel/playground/mmsegmentation/work_dirs/ocrnet_hr18_4xb4-80k_ade20k-512x512_shoe_0124/end2end_yolox_shoe.onnx";

	// initialize onnxruntime engine
	this->ortengine_shoe = new OnnxRuntimeEngine(shoe_model_path);
#endif

  // initialize onnxruntime engine
  this->ortengine_road = new OnnxRuntimeEngine(road_model_path);
  this->ortengine_border = new OnnxRuntimeEngine(border_model_path);
  this->ortengine_line = new OnnxRuntimeEngine(line_model_path);

  this->outbuffer_road = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_border = (void*)malloc(512 * 512 * sizeof(int64_t));
  this->outbuffer_line = (void*)malloc(512 * 512 * sizeof(int64_t));
};
#endif

RouteTask::~RouteTask() {
  free(outbuffer_road);
  free(outbuffer_border);
  free(outbuffer_line);

  delete (ortengine_road);
  delete (ortengine_border);
  delete (ortengine_line);
#if 0
	delete(ortengine_shoe);
#endif
}

int RouteTask::processMask(cv::Mat src, cv::Mat& finalmask, bool& hasShoe,
                           std::vector<int>& shoePos, cv::Mat& visimg) {

  int res;
  float min_mask_area_ratio = 0.03;
  clock_t startTime, endTime;

  cv::Mat final_roadmask, final_blackmask, final_whitemask, final_linemask,
      out_roadmask, out_linemask, out_linetypemask, newlabel_finalmask;
  memset(this->outbuffer_road, 0, 512 * 512 * sizeof(int64_t));
  memset(this->outbuffer_border, 0, 512 * 512 * sizeof(int64_t));
  memset(this->outbuffer_line, 0, 512 * 512 * sizeof(int64_t));

  // initialize return
  finalmask = Mat::zeros(src.rows, src.cols, CV_8UC1);

  int padw, padh;

  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image is empty\n");
    return ONNXRUNTIMEENGINE_FILE_ERROR;
  }

  startTime = clock();

  // road
  res = ortengine_road->processSeg(src, Size(512, 512), padw, padh,
                                   this->outbuffer_road);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "road model processSeg error\n");
    return res;
  }

  // road : objlabel: 1 road
  res = ortengine_road->segmentationPost(this->outbuffer_road, final_roadmask,
                                         512, 512, padw, padh, src.rows,
                                         src.cols, 1);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "road segmentationPost error\n");
    return res;
  }

  // remove small areas in road segmentation result
  FilterNoisyArea(final_roadmask, min_mask_area_ratio, out_roadmask);

  // imwrite("./out_roadmask.jpg", out_roadmask);

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

  // line
  res = ortengine_line->processSeg(src, Size(512, 512), padw, padh,
                                   this->outbuffer_line);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // road line: 1 line
  res = ortengine_line->segmentationPost(this->outbuffer_line, final_linemask,
                                         512, 512, padw, padh, src.rows,
                                         src.cols, 1);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // add final_linemask
  bitwise_and(final_linemask, final_linemask, out_linemask, out_roadmask);

  // label out_linemask to startstopline and gapline
  res = labelLineType(out_linemask, out_linetypemask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./out_linemask.jpg", out_linemask);
  cv::addWeighted(out_roadmask, 0.5, out_linetypemask, 0.5, 0, finalmask);
  cv::addWeighted(finalmask, 1.0, final_blackmask, 0.1, 0, finalmask);
  cv::addWeighted(finalmask, 1.0, final_whitemask, 0.3, 0, finalmask);

  // imwrite("./out_linemask.jpg", out_linemask);
  // imwrite("./out_roadmask.jpg", out_roadmask);

  // replace label
  res = replaceRoadMaskLabel(finalmask, newlabel_finalmask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "processSeg error\n");
    return res;
  }

  // imwrite("./newlabel_finalmask.jpg", newlabel_finalmask);

  finalmask = newlabel_finalmask;

#if 0
	// shoe detection
	std::vector<BoundingBox> shoe_outboxes;
	res = ortengine_shoe->processDet(src, Size(640, 640), 0, 0.6, shoe_outboxes);

	if (res != ONNXRUNTIMEENGINE_SUCCESS)
	{
		fprintf(stderr, "shoe model processDet error %s\n");
		return res;
	}

#if ROUTEDEBUGMODE
	visimg = src.clone();
	for (auto& box : shoe_outboxes)
	{
		ortengine_shoe->drawboxsave(visimg, (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
	}
#endif

	// Get box of shelve
	BoundingBox shoe;
	shoePos.clear();
	hasShoe = false;
	res = GetShoePos(shoe_outboxes, shoe);
	if (res != ONNXRUNTIMEENGINE_SUCCESS)
	{
		fprintf(stderr, "GetShoePos error %s\n", imageName.c_str());
		return res;
	}

	if (shoe.x2 - shoe.x1 < 1.0)
		hasShoe = false;
	else
	{
		hasShoe = true;

		shoePos.push_back((int)shoe.x1);
		shoePos.push_back((int)shoe.y1);
		shoePos.push_back((int)shoe.x2);
		shoePos.push_back((int)shoe.y2);
	}
#else
  visimg = src.clone();
#endif

  endTime = clock();
  // cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
  // << endl;

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32
int mainRouteTask() {
  // initialize onnxruntime engine
  RouteTask* routetask =
      new RouteTask(road_model_path, border_model_path, line_model_path);

  std::string filepath = "./";
  std::string outfilepath = "./";

  int imgcount = 1;
  for (int i = 0; i < imgcount; i++) {
    int res;
    Mat src, finalmask;
    std::vector<int> shoePos;
    bool hasShoe = false;
    Mat visimg;

    // std::stringstream ss;
    // ss << std::setfill('0') << std::setw(6) << i << ".jpg";
    ////ss << i << ".jpg";
    //
    // std::string imageName = ss.str();

    std::string imageName = "1704955291.502696.jpg";

    src = imread(filepath + imageName, IMREAD_COLOR); // read image

    if (src.empty()) // check image is valid
    {
      fprintf(stderr, "RouteTask Can not load image %s\n", imageName.c_str());
      continue;
    }

    printf("now process %s\n", imageName.c_str());

    res = routetask->processMask(src, finalmask, hasShoe, shoePos, visimg);
    if (res != ONNXRUNTIMEENGINE_SUCCESS) {
      fprintf(stderr, "RouteTask processMask error %s\n", imageName.c_str());
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

#if ROUTEDEBUGMODE
    // imageName is string
    std::string savepath = outfilepath +
                           imageName.substr(0, imageName.length() - 4) +
                           "_box_result.jpg";
    imwrite(savepath, visimg);

    // imageName is string
    savepath = outfilepath + imageName.substr(0, imageName.length() - 4) +
               "_result.jpg";
    imwrite(savepath, finalmask);
#endif
  }

  delete (routetask);

  return 0;
}
#else
int mainRouteTask() {
  // initialize onnxruntime engine
  RouteTask* routetask =
      new RouteTask(road_model_path, border_model_path, line_model_path);

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
    std::vector<int> shoePos;
    bool hasShoe = false;
    Mat visimg;

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
      fprintf(stderr, "RouteTask Can not load image %s\n", imageName.c_str());
      continue;
    }

    printf("now process %s\n", imageName.c_str());

    res = routetask->processMask(src, finalmask, hasShoe, shoePos, visimg);
    if (res != ONNXRUNTIMEENGINE_SUCCESS) {
      fprintf(stderr, "RouteTask processMask error %s\n", imageName.c_str());
      continue;
    }
#if ROUTEDEBUGMODE
    // imageName is string
    std::string savepath = outfilepath +
                           imageName.substr(0, imageName.length() - 4) +
                           "_box_result.jpg";
    imwrite(savepath, visimg);

    // imageName is string
    savepath = outfilepath + imageName.substr(0, imageName.length() - 4) +
               "_result.jpg";
    imwrite(savepath, finalmask);
#endif
  }

  delete (routetask);

  return 0;
}

#endif
