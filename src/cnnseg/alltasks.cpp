#include "task1forest.h"
#include "task2route.h"
#include "task3tabletnavi.h"
#ifndef _WIN32
#include "task3touchscreen.h"
#endif
#include "task4emptyseat.h"

int main() {
  mainForestTask();
  // mainRouteTask();
  // mainTabletnaviTask();
#ifndef _WIN32
  // mainTouchscreenTask();
#endif
  // mainEmptyseatTask();
}