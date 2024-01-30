#include "onnxruntimeEngine.h"

#ifndef _WIN32
#include <dirent.h>
#endif

#include <cstdio>
#include <iostream>

#include <vector>
#include "task3tabletnavi.h"

using namespace cv;
using namespace std;

#define TABLETNAVIDEBUGMODE 1

int GetShelvPos(std::vector<BoundingBox>& furn_outboxes, BoundingBox& shelve) {
  bool findobj = false;
  for (auto& box : furn_outboxes) {
    if (box.label == 0) // shelve
    {
      shelve = box;
      findobj = true;
      break;
    }
  }

  if (findobj == false) {
    shelve.label = 0;
    shelve.score = 0;
    shelve.x1 = 0;
    shelve.y1 = 0;
    shelve.x2 = 0;
    shelve.y2 = 0;
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int GetTabletPos(std::vector<BoundingBox>& furn_outboxes, BoundingBox& tablet) {
  bool findobj = false;
  for (auto& box : furn_outboxes) {
    if (box.label == 1) // tablet
    {
      tablet = box;
      findobj = true;
      break;
    }
  }

  if (findobj == false) {
    tablet.label = 0;
    tablet.score = 0;
    tablet.x1 = 0;
    tablet.y1 = 0;
    tablet.x2 = 0;
    tablet.y2 = 0;
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32

TabletnaviTask::TabletnaviTask(const wchar_t* tablet_model_path,
                               const wchar_t* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_tablet = new OnnxRuntimeEngine(tablet_model_path);

  // initialize onnxruntime engine
  foresttask = new ForestTask(border_model_path);
};

#else
TabletnaviTask::TabletnaviTask(const char* tablet_model_path,
                               const char* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_tablet = new OnnxRuntimeEngine(tablet_model_path);

  // initialize onnxruntime engine
  foresttask = new ForestTask(border_model_path);
};
#endif

TabletnaviTask::~TabletnaviTask() {
  delete (ortengine_tablet);
  delete (foresttask);
}

// available
// 0,1,2
// 3,4,5
// if available[i] = true, chair empty
int TabletnaviTask::findObject(cv::Mat src, bool& hasCabinet,
                               std::vector<int>& cabinPos, bool& hasTab,
                               std::vector<int>& tabPos, cv::Mat& bordermask,
                               cv::Mat& visimg) {
  int res;
  std::vector<BoundingBox> tablet_outboxes;

  clock_t startTime, endTime;

  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image empty\n");
    return ONNXRUNTIMEENGINE_FILE_ERROR;
  }

  startTime = clock();

  res = ortengine_tablet->processDet(src, Size(640, 640), 1, 0.6,
                                     tablet_outboxes);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "tablet model processDet error\n");
    return res;
  }

  // imageName is string
#if TABLETNAVIDEBUGMODE
  visimg = src.clone();
  for (auto& box : tablet_outboxes) {
    if (box.label == 0) // shelve
      ortengine_tablet->drawboxsave(visimg, (int)box.x1, (int)box.y1,
                                    (int)box.x2, (int)box.y2);
    else // tablet
      ortengine_tablet->drawboxsave(visimg, (int)box.x1, (int)box.y1,
                                    (int)box.x2, (int)box.y2,
                                    cv::Scalar(0, 255, 0));
  }
#endif

  // Get box of shelve
  BoundingBox shelve;
  cabinPos.clear();
  hasCabinet = false;
  res = GetShelvPos(tablet_outboxes, shelve);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "GetShelvPos error\n");
    return res;
  }

  if (shelve.x2 - shelve.x1 < 1.0)
    hasCabinet = false;
  else {
    hasCabinet = true;

    cabinPos.push_back((int)shelve.x1);
    cabinPos.push_back((int)shelve.y1);
    cabinPos.push_back((int)shelve.x2);
    cabinPos.push_back((int)shelve.y2);
  }

  // Get box of tablet
  BoundingBox tablet;
  tabPos.clear();
  hasTab = false;
  res = GetTabletPos(tablet_outboxes, tablet);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "GetTabletPos error\n");
    return res;
  }

  if (tablet.x2 - tablet.x1 < 1.0)
    hasTab = false;
  else {
    hasTab = true;

    tabPos.push_back((int)tablet.x1);
    tabPos.push_back((int)tablet.y1);
    tabPos.push_back((int)tablet.x2);
    tabPos.push_back((int)tablet.y2);
  }

  // call foresttask = border prediction
  res = foresttask->processMask(src, bordermask);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "ForestTask processMask error\n");
    return res;
  }

  endTime = clock();
  // cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
  // << endl;

  return ONNXRUNTIMEENGINE_SUCCESS;
}
