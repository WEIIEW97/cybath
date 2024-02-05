#include "onnxruntimeEngine.h"

#ifndef _WIN32
#include <dirent.h>
#endif

#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;

template <typename T>
const T& valclamp(const T& v, const T& lo, const T& hi) {
  return std::max(lo, std::min(v, hi));
}

#ifdef _WIN32
OnnxRuntimeEngine::OnnxRuntimeEngine(const wchar_t* model_path)
#else
OnnxRuntimeEngine::OnnxRuntimeEngine(const char* model_path)
#endif
{
  g_ort_rt = NULL;
  session = NULL;
  allocator = NULL;

  const OrtApi* g_ort = NULL;
  const OrtApiBase* ptr_api_base = OrtGetApiBase();
  g_ort = ptr_api_base->GetApi(ORT_API_VERSION);
  OrtEnv* env;
  CheckStatus(g_ort, g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
  OrtSessionOptions* session_options;
  CheckStatus(g_ort, g_ort->CreateSessionOptions(&session_options));
  CheckStatus(g_ort, g_ort->SetIntraOpNumThreads(session_options, 1));
  CheckStatus(g_ort, g_ort->SetSessionGraphOptimizationLevel(session_options,
                                                             ORT_ENABLE_BASIC));

#ifndef _WIN32
  // CUDA option set
  OrtCUDAProviderOptions cuda_option;
  cuda_option.device_id = 0;
  cuda_option.arena_extend_strategy = 0;
  cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  cuda_option.gpu_mem_limit = SIZE_MAX;
  cuda_option.do_copy_in_default_stream = 1;
  // CUDA acceleration
  CheckStatus(g_ort, g_ort->SessionOptionsAppendExecutionProvider_CUDA(
                         session_options, &cuda_option));
#endif

  // load  model and creat session
  // Model file path
  //  cout << "Using Onnxruntime C++ API" << endl;
  CheckStatus(g_ort,
              g_ort->CreateSession(env, model_path, session_options, &session));
  CheckStatus(g_ort, g_ort->GetAllocatorWithDefaultOptions(&allocator));

  //**********Input information**********//
  // size_t num_input_nodes; //Enter the number of nodes
  CheckStatus(g_ort, g_ort->SessionGetInputCount(session, &num_input_nodes));

  input_node_names.resize(num_input_nodes);
  input_node_dims.resize(num_input_nodes);
  input_types.resize(num_input_nodes);
  input_tensors.resize(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    // Get input node names
    char* input_name;
    CheckStatus(g_ort,
                g_ort->SessionGetInputName(session, i, allocator, &input_name));
    input_node_names[i] = input_name;

    // Get input node types
    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CheckStatus(g_ort, g_ort->GetTensorElementType(tensor_info, &type));
    input_types[i] = type;

    // Get input shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    input_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, input_node_dims[i].data(), num_dims));

    size_t tensor_size;
    CheckStatus(g_ort,
                g_ort->GetTensorShapeElementCount(tensor_info, &tensor_size));

    if (typeinfo)
      g_ort->ReleaseTypeInfo(typeinfo);
  }
  //---------------------------------------------------//

  //***********output info****************//

  CheckStatus(g_ort, g_ort->SessionGetOutputCount(session, &num_output_nodes));
  output_node_names.resize(num_output_nodes);
  output_node_dims.resize(num_output_nodes);
  output_tensors.resize(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    // Get output node names
    char* output_name;
    CheckStatus(g_ort, g_ort->SessionGetOutputName(session, i, allocator,
                                                   &output_name));
    output_node_names[i] = output_name;

    OrtTypeInfo* typeinfo;
    CheckStatus(g_ort, g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo));
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CheckStatus(g_ort, g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));

    // Get output shapes/dims
    size_t num_dims;
    CheckStatus(g_ort, g_ort->GetDimensionsCount(tensor_info, &num_dims));
    output_node_dims[i].resize(num_dims);
    CheckStatus(g_ort, g_ort->GetDimensions(
                           tensor_info, (int64_t*)output_node_dims[i].data(),
                           num_dims));

    // detection model cause crash here
    // size_t tensor_size;
    // CheckStatus(g_ort, g_ort->GetTensorShapeElementCount(tensor_info,
    // &tensor_size));

    if (typeinfo)
      g_ort->ReleaseTypeInfo(typeinfo);
  }
  //---------------------------------------------------//

  // printf("Number of inputs = %zu\n", num_input_nodes);
  // printf("Number of output = %zu\n", num_output_nodes);
  // std::cout << "input_name:" << input_node_names[0] << std::endl;
  // std::cout << "output_name: " << output_node_names[0] << std::endl;

  g_ort_rt = (OrtApi*)g_ort;
}

bool OnnxRuntimeEngine::CheckStatus(const OrtApi* g_ort, OrtStatus* status) {
  if (status != nullptr) {
    const char* msg = g_ort->GetErrorMessage(status);
    std::cerr << msg << std::endl;
    g_ort->ReleaseStatus(status);
    throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
  }
  return true;
}

int OnnxRuntimeEngine::paddingResize(cv::InputArray& src, cv::OutputArray& dst,
                                     float& fratio, int& pad_w, int& pad_h,
                                     cv::Size size, double fx, double fy,
                                     int interpolation, cv::Scalar padval) {
  float fpadd_w = 0;
  float fpadd_h = 0;
  // Here, we first find the one with the longest edge in the original image.
  // Then we resize it according to the longest edge, and fill in the smaller
  // ones.
  float r =
      std::min(float(size.width) / src.cols(), float(size.height) / src.rows());
  int inside_w = round(src.cols() * r); //
  int inside_h = round(src.rows() * r); //
  fpadd_w = size.width - inside_w;      // One of padd_w and padd_h is zero.
  fpadd_h = size.height - inside_h;
  // cout << "padd_w:" << fpadd_w << ",padd_h:" << fpadd_h << endl;
  cv::resize(src, dst, cv::Size(inside_w, inside_h), fx, fy, interpolation);

  // fpadd_w = fpadd_w / 2;
  // fpadd_h = fpadd_h / 2;
  // cout << "padd_w:" << fpadd_w << ",padd_h:" << fpadd_h << endl;
  // The round function rounds a decimal number to the nearest integer.
  // For example, round(2.2) = 2.0000; round(2.5) = 3.000.
  // Fill the outer border with white
  // int top = int(round(fpadd_h - 0.1));
  int top = 0;
  int bottom = int(round(fpadd_h + 0.1));
  // int left = int(round(fpadd_w - 0.1));
  int left = 0;
  int right = int(round(fpadd_w + 0.1));
  // cout << "top:" << top << ",bottom:" << bottom << ",left:" << left <<
  // ",right:" << right << endl;

  // top, bottom, left, right indicate the size of the expanded edges around the
  // original image
  cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT,
                     padval);

  fratio = r;
  pad_w = int(round(fpadd_w + 0.1));
  pad_h = int(round(fpadd_h + 0.1));

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::unpaddingResize(Mat& src, Mat& dst, int pad_w, int pad_h,
                                       int raw_w, int raw_h, double fx,
                                       double fy, int interpolation) {
  Mat cropimg;
  cropimg = src(Range(0, src.rows - pad_h), Range(0, src.cols - pad_w));
  cv::resize(cropimg, dst, cv::Size(raw_w, raw_h), fx, fy, interpolation);

  return ONNXRUNTIMEENGINE_SUCCESS;
}

// norm and bgr to rgb
int OnnxRuntimeEngine::preProcess(Mat& image, Mat& image_blob) {
  Mat input;
  image.copyTo(input);

  // Data processing Standardization
  std::vector<Mat> channels, channel_p;
  split(input, channels);
  Mat R, G, B;
  B = channels.at(0);
  G = channels.at(1);
  R = channels.at(2);

  B = (B - 123.675) / 58.395;
  G = (G - 116.28) / 57.12;
  R = (R - 103.53) / 57.375;

  channel_p.push_back(R);
  channel_p.push_back(G);
  channel_p.push_back(B);

  Mat outt;
  merge(channel_p, outt);
  image_blob = outt;

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::processSeg(Mat& img, Size targetSize, int& padw,
                                  int& padh, void* output_buffer) {
  int res;
  const OrtApi* g_ort = (const OrtApi*)g_ort_rt;

  // preprocess norm and bgr to rgb
  Mat dst, preprocimg;
  img.convertTo(img, CV_32FC3);
  preProcess(img, dst);

  // padding and resize
  // int padw, padh;
  float fratio;
  void* local_output_buffer = NULL;

  res = paddingResize(dst, preprocimg, fratio, padw, padh, targetSize, 0, 0,
                      cv::INTER_LINEAR);
  if (res != ONNXRUNTIMEENGINE_SUCCESS)
    return res;

  // Load image
  Mat blob =
      dnn::blobFromImage(preprocimg, 1., Size(preprocimg.rows, preprocimg.cols),
                         Scalar(0, 0, 0), false, true);
  size_t input_data_length = blob.total() * blob.elemSize();
  // printf("Load success!\n");

  // Create an input tensor
  OrtMemoryInfo* memory_info;

  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
                         memory_info, reinterpret_cast<void*>(blob.data),
                         input_data_length, input_node_dims[0].data(),
                         input_node_dims[0].size(), input_types[0],
                         &input_tensors[0]));

  g_ort->ReleaseMemoryInfo(memory_info);

  CheckStatus(g_ort,
              g_ort->Run(session, nullptr, input_node_names.data(),
                         (const OrtValue* const*)input_tensors.data(),
                         input_tensors.size(), output_node_names.data(),
                         output_node_names.size(), output_tensors.data()));

  local_output_buffer = NULL;
  CheckStatus(g_ort, g_ort->GetTensorMutableData(output_tensors[0],
                                                 &local_output_buffer));

  memcpy(output_buffer, local_output_buffer,
         preprocimg.rows * preprocimg.cols * sizeof(int64_t));

  // Mat finalmask;
  //
  // res = segmentationPost(local_output_buffer, finalmask, preprocimg.rows,
  // preprocimg.cols, padw, padh, img.rows, img.cols, 1); if (res !=
  // ONNXRUNTIMEENGINE_SUCCESS) 	return res;

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::processDet(Mat& img, Size targetSize,
                                  long long maxclassid, float threshold,
                                  std::vector<BoundingBox>& outboxes) {
  int res;
  const OrtApi* g_ort = (const OrtApi*)g_ort_rt;

  // imwrite("./raw.jpg", img);

  // preprocess norm and bgr to rgb
  Mat preprocimg;
  img.convertTo(img, CV_32FC3);

  // padding and resize
  int padw, padh;
  float fratio;
  void* dets_buffer = NULL;
  void* labels_buffer = NULL;

  res = paddingResize(img, preprocimg, fratio, padw, padh, targetSize, 0, 0,
                      cv::INTER_LINEAR, cv::Scalar(114.0, 114.0, 114.0));
  if (res != ONNXRUNTIMEENGINE_SUCCESS)
    return res;

  // imwrite("./padding.jpg", preprocimg);

  // Load image
  Mat blob =
      dnn::blobFromImage(preprocimg, 1., Size(preprocimg.rows, preprocimg.cols),
                         Scalar(0, 0, 0), false, true);
  size_t input_data_length = blob.total() * blob.elemSize();
  // printf("Load success!\n");

  // Create an input tensor
  OrtMemoryInfo* memory_info;

  CheckStatus(g_ort, g_ort->CreateCpuMemoryInfo(
                         OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
  CheckStatus(g_ort, g_ort->CreateTensorWithDataAsOrtValue(
                         memory_info, reinterpret_cast<void*>(blob.data),
                         input_data_length, input_node_dims[0].data(),
                         input_node_dims[0].size(), input_types[0],
                         &input_tensors[0]));

  g_ort->ReleaseMemoryInfo(memory_info);

  CheckStatus(g_ort,
              g_ort->Run(session, nullptr, input_node_names.data(),
                         (const OrtValue* const*)input_tensors.data(),
                         input_tensors.size(), output_node_names.data(),
                         output_node_names.size(), output_tensors.data()));

  dets_buffer = NULL;
  CheckStatus(g_ort,
              g_ort->GetTensorMutableData(output_tensors[0], &dets_buffer));

  labels_buffer = NULL;
  CheckStatus(g_ort,
              g_ort->GetTensorMutableData(output_tensors[1], &labels_buffer));

  res = detectionPost(dets_buffer, labels_buffer, outboxes, fratio, 100,
                      maxclassid, img.cols, img.rows, threshold);
  if (res != ONNXRUNTIMEENGINE_SUCCESS)
    return res;

  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::classificationPost(void* output_buffer,
                                          size_t output_data_size) {
  float* float_buffer = reinterpret_cast<float*>(output_buffer);
  // size_t output_data_size = output_node_dims[0][1];
  auto max = std::max_element(float_buffer, float_buffer + output_data_size);

  std::vector<float> optu(float_buffer, float_buffer + output_data_size);
  int max_index = static_cast<int>(std::distance(float_buffer, max));

  // Loading label file
  std::fstream label_file(
      "D:\\xx\\onnxruntimetest\\classification_classes_ILSVRC2012.txt",
      std::ios::in);
  std::unordered_map<int, std::string> label_table;
  label_table.reserve(output_data_size);
  int i = 0;
  std::string line;
  while (std::getline(label_file, line)) {
    label_table.emplace(i++, line);
  }

  // printf("%d, %f, %s \n", max_index, *max, label_table[max_index].c_str());
  // printf("Done!\n");
  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::segmentationPost(void* output_buffer, Mat& mask,
                                        int exrows, int excols, int expadw,
                                        int expadh, int rawrows, int rawcols,
                                        int64_t objlabel) {
  // Get pointer to output tensor float values
  int64_t* intarr = reinterpret_cast<int64_t*>(output_buffer);

  Mat exmask = Mat_<unsigned char>(exrows, excols); //

  int idx = 0;
  for (int i = 0; i < exmask.rows; i++) {
    for (int j = 0; j < exmask.cols; j++) {
      int64_t pixel = intarr[idx++];

      if (pixel == objlabel)
        exmask.at<unsigned char>(i, j) = (unsigned char)(255);
      else
        exmask.at<unsigned char>(i, j) = 0;
    }
  }

  // imwrite("./exmask.jpg", exmask);

  //
  Mat unpadmask;
  unpaddingResize(exmask, unpadmask, expadw, expadh, rawcols, rawrows, 0, 0,
                  cv::INTER_LINEAR);

  threshold(unpadmask, mask, 128, 255, 0);

  // std::cout << "Done!" << std::endl;
  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::detectionPost(void* dets_buffer, void* labels_buffer,
                                     std::vector<BoundingBox>& outboxes,
                                     float fratio, int outboxnum,
                                     long long maxclassid, int rawcols,
                                     int rawrows, float threshold) {
  // Get pointer to output tensor float values
  float* floatarr = reinterpret_cast<float*>(dets_buffer);
  long long* intarr = reinterpret_cast<long long*>(labels_buffer);

  int idx1 = 0;
  int idx2 = 0;
  bool firstdraw = true;
  for (int i = 0; i < outboxnum; i++) {
    BoundingBox box;
    box.label = (long long)(intarr[idx1++]);

    box.x1 = valclamp((float)((floatarr[idx2++]) / fratio), (float)0.0,
                      (float)(rawcols - 1));
    box.y1 = valclamp((float)((floatarr[idx2++]) / fratio), (float)0.0,
                      (float)(rawrows - 1));
    box.x2 = valclamp((float)((floatarr[idx2++]) / fratio), (float)0.0,
                      (float)(rawcols - 1));
    box.y2 = valclamp((float)((floatarr[idx2++]) / fratio), (float)0.0,
                      (float)(rawrows - 1));
    box.score = (float)(floatarr[idx2++]);

    // for debug
    // if (box.label == 0)
    //	continue;

    if (box.label > maxclassid)
      continue;

    if (box.score < threshold)
      continue;

    // if (firstdraw)
    //{
    //	drawboxsave("./000000.jpg", "./image_with_rectangle.jpg", (int)box.x1,
    //(int)box.y1, (int)box.x2, (int)box.y2); 	firstdraw = false;
    // }
    // else
    //{
    //	drawboxsave("./image_with_rectangle.jpg", "./image_with_rectangle.jpg",
    //(int)box.x1, (int)box.y1, (int)box.x2, (int)box.y2);
    // }

    outboxes.push_back(box);
  }

  // std::cout << "Done!" << std::endl;
  return ONNXRUNTIMEENGINE_SUCCESS;
}

int OnnxRuntimeEngine::drawboxsave(Mat& src, int x1, int y1, int x2, int y2,
                                   cv::Scalar color) {
  // position
  cv::Point pt1(x1, y1);
  cv::Point pt2(x2, y2);

  cv::rectangle(src, pt1, pt2, color, 2);

  return ONNXRUNTIMEENGINE_SUCCESS;
}