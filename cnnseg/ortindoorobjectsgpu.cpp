#include "onnxruntimeEngine.h"

#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

#include <vector>
#include <algorithm>
#include "ortobjectgpu.h"

using namespace cv;
using namespace std;

#define INDOOROBJS_ERR_BASE 100
#define INDOOROBJS_ERR_NO_VALIBLE_SEATS 101
#define DEBUGMODE 0

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

// FIX: chair not detected because occuluded by person
// TODO: bag box overlap with two chair box
// leftview = true: chair on left
// false: chair on right
int FindAvailableChair(std::vector<BoundingBox>& person_outboxes, std::vector<BoundingBox>& furn_outboxes, bool available[6], bool leftview, bool sort_with_person = false)
{
	std::vector<BoundingBox> chairsboxs;
	std::vector<BoundingBox> bagboxs;

	for (int i = 0; i < 6; i++)
		available[i] = true;

	for (auto& box : furn_outboxes)
	{
		if (box.label == 0) // chair
			chairsboxs.push_back(box);

		if (box.label == 1) // bag
			bagboxs.push_back(box);
	}

	// FIX: chair not detected because occuluded by person
	// append person boxes, be careful, person may in background area
	if (sort_with_person)
	{
		for (auto& box : person_outboxes)
		{
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

	for (auto& box : chairsboxs)
	{
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
	// object in upper part of chair, top of bag's distance to top of chairbox is less than 1/3 of bag height
	// first row chairs

	for (auto& bagbox : bagboxs)
	{
		int idx = 0;
		int maxarea = (int)(boxarea(bagbox) * 0.85);
		int maxid = -1;
		for (auto& chairbox : top_chairsboxs)
		{
			int overlap = overlapArea(bagbox, chairbox);

			if (overlap > maxarea && abs(min(bagbox.y1, bagbox.y2) - chairbox.y1) < abs((bagbox.y2 - bagbox.y1) * 1.0 / 3))
			{
				maxarea = overlap;
				maxid = idx;
			}
			idx++;
		}

		if (maxid >= 0)
		{
			available[maxid] = false;
		}
	}

	// second row chairs
	for (auto& bagbox : bagboxs)
	{
		int idx = 0;
		int maxarea = (int)(boxarea(bagbox) * 0.85);
		int maxid = -1;
		for (auto& chairbox : bottom_chairsboxs)
		{
			int overlap = overlapArea(bagbox, chairbox);

			if (overlap > maxarea && abs(min(bagbox.y1, bagbox.y2) - chairbox.y1) < abs((bagbox.y2 - bagbox.y1) * 1.0 / 3))
			{
				maxarea = overlap;
				maxid = idx;
			}
			idx++;
		}

		if (maxid >= 0)
		{
			available[maxid + 3] = false;
		}
	}


	// the most overlapped chair of person
	// personbox is added in chair box, then must higly overlap
	// personbox is not added, will not meet this maxarea, because no occuluded chairs detected
	// becareful, the person in background may introduce unwanted result
	for (auto& personbox : person_outboxes)
	{
		int idx = 0;
		int maxarea = (int)(boxarea(personbox) * 0.95);
		int maxid = -1;
		for (auto& chairbox : chairsboxs)
		{
			int overlap = overlapArea(personbox, chairbox);
			if (overlap > maxarea)
			{
				maxarea = overlap;
				maxid = idx;
			}
			idx++;
		}

		if (maxid >= 0)
		{
			available[maxid] = false;
		}
	}

	return ONNXRUNTIMEENGINE_SUCCESS;
}

int GetShelvPos(std::vector<BoundingBox>& furn_outboxes, BoundingBox& shelve)
{
	bool findobj = false;
	for (auto& box : furn_outboxes)
	{
		if (box.label == 2) // shelve
		{
			shelve = box;
			findobj = true;
			break;
		}
	}

	if (findobj == false)
	{
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

ortObjectGPU::ortObjectGPU()
{
	const wchar_t* coco_model_path = L"C:/Users/Administrator/Desktop/end2end_yolox_coco.onnx";
	const wchar_t* furnitures_model_path = L"C:/Users/Administrator/Desktop/end2end_yolox_furnitures.onnx";

	// initialize onnxruntime engine
	this->ortengine_coco = new OnnxRuntimeEngine(coco_model_path);
	this->ortengine_furn = new OnnxRuntimeEngine(furnitures_model_path);
};

#else
ortObjectGPU::ortObjectGPU()
{
	const char* coco_model_path = "/algdata01/huan.wang/samlabel/playground/mmdetection/work_dirs/yolox_s_8xb8-300e_coco/end2end_yolox_coco.onnx";
	const char* furnitures_model_path = "/algdata01/huan.wang/samlabel/playground/mmdetection/work_dirs/yolox_s_8xb8-300e_coco/end2end_yolox_furnitures.onnx";

	// initialize onnxruntime engine
	this->ortengine_coco = new OnnxRuntimeEngine(coco_model_path);
	this->ortengine_furn = new OnnxRuntimeEngine(furnitures_model_path);
};
#endif

ortObjectGPU::~ortObjectGPU()
{
	delete(ortengine_coco);
	delete(ortengine_furn);
}

// available 
// 0,1,2
// 3,4,5
// if available[i] = true, chair empty
int ortObjectGPU::findChair(cv::Mat src, bool available[6], bool& hasCabinet, std::vector<int>& position, std::string imageName)
{
	int res;
	std::vector<BoundingBox> coco_outboxes, furn_outboxes;

	clock_t startTime, endTime;

	if (src.empty())     // check image is valid  
	{
		fprintf(stderr, "src image empty %s\n", imageName.c_str());
		return ONNXRUNTIMEENGINE_FILE_ERROR;
	}

	startTime = clock();

	// only get person label
	res = ortengine_coco->processDet(src, Size(640, 640), 0, 0.5, coco_outboxes);

	if (res != ONNXRUNTIMEENGINE_SUCCESS)
	{
		fprintf(stderr, "processSeg error %s\n", imageName.c_str());
		return res;
	}

	res = ortengine_furn->processDet(src, Size(640, 640), 2, 0.6, furn_outboxes);

	if (res != ONNXRUNTIMEENGINE_SUCCESS)
	{
		fprintf(stderr, "processSeg error %s\n", imageName.c_str());
		return res;
	}

	// imageName is string
#ifdef _WIN32
	std::string filepath = "./";
	std::string outfilepath = "./";
#else
	std::string filepath = "/algdata01/huan.wang/samlabel/labelme/examples/instance_segmentation/furnitures/chairsshelves_coco/JPEGImages/";
	std::string outfilepath = "/algdata01/huan.wang/samlabel/playground/mmdetection/data/my_set1/result/";
#endif

#if DEBUGMODE
	std::string savepath = outfilepath + imageName.substr(0, imageName.length() - 4) + "_result.jpg";

	bool firstsave = true;
	for (auto& box : coco_outboxes)
	{
		if (firstsave)
		{
			ortengine_coco->drawboxsave(filepath + imageName, savepath, (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
			firstsave = false;
		}
		else
		{
			ortengine_coco->drawboxsave(savepath, savepath, (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
		}
	}

	for (auto& box : furn_outboxes)
	{
		if (firstsave)
		{
			ortengine_furn->drawboxsave(filepath + imageName, savepath, (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
			firstsave = false;
		}
		else
		{
			ortengine_furn->drawboxsave(savepath, savepath, (int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
		}
	}
#endif

	// not equal to 6 chairs
	int count_chair = 0;
	for (auto& box : furn_outboxes)
	{
		if (box.label == 0) // chair
			count_chair++;
	}
	if (count_chair == 6)
	{
		// find available seats, if available[i] == true
		for (int idx = 0; idx < 6; idx++)
			available[idx] = true;

		// 
		res = FindAvailableChair(coco_outboxes, furn_outboxes, available, true);
		if (res != ONNXRUNTIMEENGINE_SUCCESS)
		{
			for (int idx = 0; idx < 6; idx++)
				available[idx] = false;

			fprintf(stderr, "processSeg error %s\n", imageName.c_str());
			return res;
		}

	}
	else
	{
		// process !=6 chairs
		fprintf(stderr, "cannot find 6 chairs\n", imageName.c_str());

		for (int idx = 0; idx < 6; idx++)
			available[idx] = false;
	}

	// Get box of shelve
	BoundingBox shelve;
	hasCabinet = false;
	res = GetShelvPos(furn_outboxes, shelve);
	if (res != ONNXRUNTIMEENGINE_SUCCESS)
	{
		fprintf(stderr, "processDet error %s\n", imageName.c_str());
		return res;
	}

	if (shelve.x2 - shelve.x1 < 1.0)
		hasCabinet = false;
	else
	{
		hasCabinet = true;

		position.push_back((int)shelve.x1);
		position.push_back((int)shelve.y1);
		position.push_back((int)shelve.x2);
		position.push_back((int)shelve.y2);
	}


	endTime = clock();
	// cout << i << ": " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

	return ONNXRUNTIMEENGINE_SUCCESS;
}

#ifdef _WIN32
int main()
{
	ortObjectGPU* ort_object_gpu = new ortObjectGPU();

	std::string filepath = "./";

	int imgcount = 1;
	//int imgcount = 917;
	for (int i = 0; i < imgcount; i++)
	{
		int res;
		cv::Mat src;
		std::vector<int> position;
		bool available[6] = { false };
		bool hasCabinet = false;

		//std::stringstream ss;
		//ss << std::setfill('0') << std::setw(6) << i << ".jpg";
		////ss << i << ".jpg";
		//
		//std::string imageName = ss.str();

		std::string imageName = "000075.jpg";

		src = imread(filepath + imageName, IMREAD_COLOR);   // read image 

		if (src.empty())     // check image is valid  
		{
			fprintf(stderr, "Can not load image %s\n", imageName.c_str());
			continue;
		}

		// only get person label
		res = ort_object_gpu->findChair(src, available, hasCabinet, position, imageName);

		if (res != ONNXRUNTIMEENGINE_SUCCESS)
		{
			fprintf(stderr, "processSeg error %s\n", imageName.c_str());
			continue;
		}

#if DEBUGMODE
		for (int i = 0; i < 6; i++)
		{
			if (available[i])
				std::cout << "seat row " << floor(i / 3) + 1 << " col " << i % 3 + 1 << " is available." << std::endl;
		}

		if (hasCabinet)
			std::cout << "shelve at (" << position[0] << "," << position[1] << "), (" << position[2] << "," << position[3] << ")" << std::endl;
#endif


	}

	delete(ort_object_gpu);

	return 0;

}
#else
int main()
{
	ortObjectGPU* ort_object_gpu = new ortObjectGPU();

	std::string filepath = "/algdata01/huan.wang/samlabel/labelme/examples/instance_segmentation/furnitures/chairsshelves_coco/JPEGImages/";

	DIR* dp = nullptr;
	const std::string& exten = "*";
	struct dirent* dirp = nullptr;
	if ((dp = opendir(filepath.c_str())) == nullptr) {
		fprintf(stderr, "scandir error %s\n", filepath.c_str());
		return ONNXRUNTIMEENGINE_PATH_ERROR;
	}

	int i = 0;
	while ((dirp = readdir(dp)) != nullptr)
	{
		int res;
		cv::Mat src;
		std::vector<int> position;
		bool available[6] = { false };
		bool hasCabinet = false;

		std::string imageName;

		if (dirp->d_type == DT_REG) {
			if (exten.compare("*") == 0)
				imageName = dirp->d_name;
			else
				if (std::string(dirp->d_name).find(exten) != std::string::npos)
					imageName = dirp->d_name;
		}
		else
			continue;

		src = imread(filepath + imageName, IMREAD_COLOR);   // read image

		if (src.empty())     // check image is valid
		{
			fprintf(stderr, "Can not load image %s\n", imageName.c_str());
			continue;
		}

		printf("now process %s\n", imageName.c_str());

		// only get person label
		res = ort_object_gpu->findChair(src, available, hasCabinet, position, imageName);

		if (res != ONNXRUNTIMEENGINE_SUCCESS)
		{
			fprintf(stderr, "processSeg error %s\n", imageName.c_str());
			continue;
		}

#if DEBUGMODE
		for (int i = 0; i < 6; i++)
		{
			if (available[i])
				std::cout << "seat row " << floor(i / 3) + 1 << " col " << i % 3 + 1 << " is available." << std::endl;
		}

		if (hasCabinet)
			std::cout << "shelve at (" << position[0] << "," << position[1] << "), (" << position[2] << "," << position[3] << ")" << std::endl;
#endif

	}

	closedir(dp);
	delete(ort_object_gpu);

	return 0;

}

#endif