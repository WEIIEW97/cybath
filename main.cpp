#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/connected_component.h"
#include "src/estimate.h"

using namespace std;

int main(int, char**){
    string pgm_path = "/home/william/Codes/find-landmark/test.pgm";
    auto pgm = cv::imread(pgm_path, -1);
    auto exit_point = estimate_trajectory(pgm, 254, 15);
    cv::Mat pgm_flip;
    cv::flip(pgm, pgm_flip, 0);
    for(auto& p : exit_point) {
      cv::circle(pgm_flip, p, 2, cv::Scalar(125, 0, 0), 2);
    }
    cv::imshow("end point", pgm_flip);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
