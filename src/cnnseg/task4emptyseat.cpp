#include "onnxruntimeEngine.h"

#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

#include <vector>
#include <algorithm>
#include "task4emptyseat.h"
#include "common.h"

using namespace cv;
using namespace std;

#define INDOOROBJS_ERR_BASE             100
#define INDOOROBJS_ERR_NO_VALIBLE_SEATS 101
#define EMPTYSEATDEBUGMODE              1

// FIX: chair not detected because occuluded by person
// TODO: bag box overlap with two chair box
// leftview = true: chair on left
// false: chair on right
int FindAvailableChair(std::vector<BoundingBox>& person_outboxes,
                       std::vector<BoundingBox>& furn_outboxes,
                       bool available[6], bool leftview,
                       bool sort_with_person = false) {
  std::vector<BoundingBox> chairsboxs;
  std::vector<BoundingBox> bagboxs;

  for (int i = 0; i < 6; i++)
    available[i] = true;

  for (auto& box : furn_outboxes) {
    if (box.label == 0) // chair
      chairsboxs.push_back(box);

    if (box.label == 1) // bag
      bagboxs.push_back(box);
  }

  // FIX: chair not detected because occuluded by person
  // append person boxes, be careful, person may in background area
  if (sort_with_person) {
    for (auto& box : person_outboxes) {
      // person box top - 1/2 head to align with chair height
      box.y1 += floor((box.y2 - box.y1) * 1.0 / 3);
      chairsboxs.push_back(box);
    }
  }

  // sort chair boxs by x
  std::sort(chairsboxs.begin(), chairsboxs.end(), xcompare);

  // if chairs at left side
  // bottom row's top is the leftmost chair top
  // if chairs at right side
  // bottom row's top is the leftmost chair top
  int bottomrow_topest;

  if (leftview)
    // chairs at left side
    bottomrow_topest = chairsboxs[0].y1;
  else
    // chairs at right side
    bottomrow_topest = chairsboxs[chairsboxs.size() - 1].y1;

  // supose 2xN
  std::vector<BoundingBox> top_chairsboxs;
  std::vector<BoundingBox> bottom_chairsboxs;

  for (auto& box : chairsboxs) {
    if (box.y1 < bottomrow_topest)
      top_chairsboxs.push_back(box);
    else
      bottom_chairsboxs.push_back(box);
  }

  std::sort(top_chairsboxs.begin(), top_chairsboxs.end(), xcompare);
  std::sort(bottom_chairsboxs.begin(), bottom_chairsboxs.end(), xcompare);

  if (top_chairsboxs.size() != 3 || bottom_chairsboxs.size() != 3)
    return INDOOROBJS_ERR_NO_VALIBLE_SEATS;

  // check max iou of bag, person with chairs
  // maxrea control the overlap ratio of bag
  // object in upper part of chair, top of bag's distance to top of chairbox is
  // less than 1/3 of bag height first row chairs

  for (auto& bagbox : bagboxs) {
    int idx = 0;
    int maxarea = (int)(boxarea(bagbox) * 0.85);
    int maxid = -1;
    for (auto& chairbox : top_chairsboxs) {
      int overlap = overlapArea(bagbox, chairbox);

      if (overlap > maxarea && abs(min(bagbox.y1, bagbox.y2) - chairbox.y1) <
                                   abs((bagbox.y2 - bagbox.y1) * 1.0 / 3)) {
        maxarea = overlap;
        maxid = idx;
      }
      idx++;
    }

    if (maxid >= 0) {
      available[maxid] = false;
    }
  }

  // second row chairs
  for (auto& bagbox : bagboxs) {
    int idx = 0;
    int maxarea = (int)(boxarea(bagbox) * 0.85);
    int maxid = -1;
    for (auto& chairbox : bottom_chairsboxs) {
      int overlap = overlapArea(bagbox, chairbox);

      if (overlap > maxarea && abs(min(bagbox.y1, bagbox.y2) - chairbox.y1) <
                                   abs((bagbox.y2 - bagbox.y1) * 1.0 / 3)) {
        maxarea = overlap;
        maxid = idx;
      }
      idx++;
    }

    if (maxid >= 0) {
      available[maxid + 3] = false;
    }
  }

  // the most overlapped chair of person
  // personbox is added in chair box, then must higly overlap
  // personbox is not added, will not meet this maxarea, because no occuluded
  // chairs detected becareful, the person in background may introduce unwanted
  // result
  for (auto& personbox : person_outboxes) {
    int idx = 0;
    int maxarea = (int)(boxarea(personbox) * 0.95);
    int maxid = -1;
    for (auto& chairbox : chairsboxs) {
      int overlap = overlapArea(personbox, chairbox);
      if (overlap > maxarea) {
        maxarea = overlap;
        maxid = idx;
      }
      idx++;
    }

    if (maxid >= 0) {
      available[maxid] = false;
    }
  }

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int GetLowShelvPos(std::vector<BoundingBox>& furn_outboxes,
                   BoundingBox& shelve) {
  bool findobj = false;
  for (auto& box : furn_outboxes) {
    if (box.label == 2) // shelve
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

#ifdef _WIN32

EmptyseatTask::EmptyseatTask(const wchar_t* coco_model_path,
                             const wchar_t* furnitures_model_path,
                             const wchar_t* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_coco = new OnnxRuntimeEngine(coco_model_path);
  this->ortengine_furn = new OnnxRuntimeEngine(furnitures_model_path);

  // initialize onnxruntime engine
  foresttask = new ForestTask(border_model_path);
};

#else
EmptyseatTask::EmptyseatTask(const char* coco_model_path,
                             const char* furnitures_model_path,
                             const char* border_model_path) {
  // initialize onnxruntime engine
  this->ortengine_coco = new OnnxRuntimeEngine(coco_model_path);
  this->ortengine_furn = new OnnxRuntimeEngine(furnitures_model_path);

  // initialize onnxruntime engine
  foresttask = new ForestTask(border_model_path);
};
#endif

EmptyseatTask::~EmptyseatTask() {
  delete (ortengine_coco);
  delete (ortengine_furn);

  delete (foresttask);
}

// available
// 0,1,2
// 3,4,5
// if available[i] = true, chair empty
int EmptyseatTask::findChair(cv::Mat src, bool available[6], bool& hasCabinet,
                             std::vector<int>& position, cv::Mat& bordermask,
                             cv::Mat& visimg) {
  int res;
  std::vector<BoundingBox> coco_outboxes, furn_outboxes;

  clock_t startTime, endTime;

  if (src.empty()) // check image is valid
  {
    fprintf(stderr, "src image empty\n");
    return ONNXRUNTIMEENGINE_FILE_ERROR;
  }

  startTime = clock();

  // only get person label
  res = ortengine_coco->processDet(src, Size(640, 640), 0, 0.5, coco_outboxes);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "coco model processDet error\n");
    return res;
  }

  res = ortengine_furn->processDet(src, Size(640, 640), 2, 0.6, furn_outboxes);

  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "furniture model processDet error\n");
    return res;
  }

  // imageName is string
#if EMPTYSEATDEBUGMODE
  bool firstsave = true;

  visimg = src.clone();
  for (auto& box : coco_outboxes) {
    ortengine_coco->drawboxsave(visimg, (int)box.x1, (int)box.y1, (int)box.x2,
                                (int)box.y2, cv::Scalar(255, 0, 0));
  }

  for (auto& box : furn_outboxes) {
    if (box.label == 0) // chair
      ortengine_furn->drawboxsave(visimg, (int)box.x1, (int)box.y1, (int)box.x2,
                                  (int)box.y2);
    else if (box.label == 1) // bug
      ortengine_furn->drawboxsave(visimg, (int)box.x1, (int)box.y1, (int)box.x2,
                                  (int)box.y2, cv::Scalar(0, 255, 0));
    else // lowshelve
      ortengine_furn->drawboxsave(visimg, (int)box.x1, (int)box.y1, (int)box.x2,
                                  (int)box.y2, cv::Scalar(255, 255, 0));
  }
#endif

  // not equal to 6 chairs
  int count_chair = 0;
  for (auto& box : furn_outboxes) {
    if (box.label == 0) // chair
      count_chair++;
  }
  if (count_chair == 6) {
    // find available seats, if available[i] == true
    for (int idx = 0; idx < 6; idx++)
      available[idx] = true;

    //
    res = FindAvailableChair(coco_outboxes, furn_outboxes, available, true);
    if (res != ONNXRUNTIMEENGINE_SUCCESS) {
      for (int idx = 0; idx < 6; idx++)
        available[idx] = false;

      fprintf(stderr, "FindAvailableChair error \n");
      return res;
    }

  } else {
    // process !=6 chairs
    fprintf(stderr, "cannot find 6 chairs\n");

    for (int idx = 0; idx < 6; idx++)
      available[idx] = false;
  }

  // Get box of lowshelve
  BoundingBox lowshelve;
  hasCabinet = false;
  position.clear();
  res = GetLowShelvPos(furn_outboxes, lowshelve);
  if (res != ONNXRUNTIMEENGINE_SUCCESS) {
    fprintf(stderr, "GetLowShelvPos error %s");
    return res;
  }

  if (lowshelve.x2 - lowshelve.x1 < 1.0)
    hasCabinet = false;
  else {
    hasCabinet = true;

    position.push_back((int)lowshelve.x1);
    position.push_back((int)lowshelve.y1);
    position.push_back((int)lowshelve.x2);
    position.push_back((int)lowshelve.y2);
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