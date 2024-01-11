#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/connected_component.h"
#include "src/estimate.h"

using namespace std;

int main(int, char**){
    string pgm_path = "/home/william/Codes/find-landmark/test.pgm";
    auto pgm = cv::imread(pgm_path, -1);
    auto exit_point1 = estimate_trajectory(pgm, 254, 15).back();
    auto exit_point2 = get_wavefront_exit(pgm, cv::Point(31, 5), 254, 15);
    cout << exit_point1 << ", " << exit_point2 << endl;
    cv::Mat pgm_flip;
    cv::flip(pgm, pgm_flip, 0);
//    for(auto& p : exit_point) {
      cv::circle(pgm_flip, exit_point1, 2, cv::Scalar(125, 0, 0), 2);
      cv::circle(pgm_flip, exit_point2, 2, cv::Scalar(77, 0, 0), 2);
//    }
    cv::imshow("end point", pgm_flip);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
