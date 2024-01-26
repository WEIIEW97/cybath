#include "common.h"

using namespace cv;
using namespace std;



bool labelareacmp(LabelArea x, LabelArea y)
{
	return x.area > y.area;
}

int FilterNoisyArea(Mat& mask, float min_mask_area_ratio, Mat& outmask)
{
	// find connectedcomponents
	Mat labels, stats, centroids;
	int num_labels = connectedComponentsWithStats(mask, labels, stats, centroids, 8);

	if (num_labels < 1)
	{
		fprintf(stderr, "no object!\n");
		return ONNXRUNTIMEENGINE_NO_MASK;
	}

	// label in labellist is ordered by area 
	std::vector<LabelArea> labellist((int64)num_labels - 1);
	for (int i = 1; i < num_labels; i++)
	{
		labellist[(int64)i - 1].label = i;
		labellist[(int64)i - 1].area = stats.at<int>(i, CC_STAT_AREA);
	}

	sort(labellist.begin(), labellist.end(), labelareacmp);

	int maskW = mask.cols;
	int maskH = mask.rows;
	Mat image_filtered = Mat::zeros(maskH, maskW, CV_8UC1);

	int min_mask_area = int(min_mask_area_ratio * maskH * maskW);
	// printf("min_mask_area: %d\n", min_mask_area);

	// filter area smaller than min_mask_area
	std::vector<Vec3b> colors(num_labels);
	for (int i = 0; i < num_labels; i++)
		colors[i] = Vec3b(0, 0, 0);

	for (int i = 1; i < num_labels; i++)
	{
		int label = labellist[(int64)i - 1].label;

		int maskArea = stats.at<int>(label, CC_STAT_AREA);

		// printf("maskArea: %d\n", maskArea);

		if (maskArea > min_mask_area)
			colors[label] = Vec3b(255, 255, 255);
		else
			colors[label] = Vec3b(0, 0, 0);

		// label in labellist is ordered by area
		// so here only process the largest area
		break;

	}

	for (int y = 0; y < image_filtered.rows; y++)
	{
		for (int x = 0; x < image_filtered.cols; x++)
		{
			int label = labels.at<int>(y, x);
			// CV_Assert(0 <= label && label <= num_labels);
			if (0 > label || label > num_labels)
			{
				fprintf(stderr, "label exceed [%d, %d]\n", 0, num_labels);
				return ONNXRUNTIMEENGINE_NO_MASK;
			}

			image_filtered.at<uchar>(y, x) = uchar(colors[label][0]);

		}
	}

	// imwrite("image_filtered.jpg", image_filtered);
	bitwise_and(mask, mask, outmask, mask = image_filtered);

	return ONNXRUNTIMEENGINE_SUCCESS;
}


// sort x first , then y
bool xycompare(const BoundingBox& a, const BoundingBox& b) {
	if (a.x1 < b.x1) return true;
	if (a.x1 > b.x1) return false;
	return a.y1 < b.y1;
}

bool xcompare(const BoundingBox& a, const BoundingBox& b) {
	return a.x1 < b.x1;
}

bool ycompare(const BoundingBox& a, const BoundingBox& b) {
	return a.y1 < b.y1;
}

int overlapArea(const BoundingBox& r1, const BoundingBox& r2) {
	int left = std::max(r1.x1, r2.x1);
	int right = std::min(r1.x2, r2.x2);
	int top = std::max(r1.y1, r2.y1);
	int bottom = std::min(r1.y2, r2.y2);

	int width = std::max(0, right - left);
	int height = std::max(0, bottom - top);

	return width * height;
}

int boxarea(const BoundingBox& r1)
{
	return (r1.x2 - r1.x1) * (r1.y2 - r1.y1);
}