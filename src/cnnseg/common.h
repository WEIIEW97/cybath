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

#endif
