#include "../src/cnnseg/onnxruntimeEngine.h"
#include <vector>

class ortObjectGPU {


    public:
        ortObjectGPU();

        ~ortObjectGPU();

        int findChair(cv::Mat src, bool available[6], bool& hasCabinet, std::vector<int>& position, std::string imageName = "nvp.jpg");
    
    private:
        OnnxRuntimeEngine* ortengine_coco;
        OnnxRuntimeEngine* ortengine_furn;
};