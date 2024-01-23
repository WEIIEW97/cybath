// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>

#include <fstream>
#include <iostream>
#include <thread>
#include <string>
#include <vector>
// #include <cuda_runtime_api.h>
#include "opencv2/opencv.hpp"
// #include <unistd.h>

using namespace cv;
using namespace std;

#ifdef HAVE_TENSORRT_PROVIDER_FACTORY_H
#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>

std::unique_ptr<OrtTensorRTProviderOptionsV2>
get_default_trt_provider_options() {
  auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
  tensorrt_options->device_id = 0;
  tensorrt_options->has_user_compute_stream = 0;
  tensorrt_options->user_compute_stream = nullptr;
  tensorrt_options->trt_max_partition_iterations = 1000;
  tensorrt_options->trt_min_subgraph_size = 1;
  tensorrt_options->trt_max_workspace_size = 1 << 30;
  tensorrt_options->trt_fp16_enable = false;
  tensorrt_options->trt_int8_enable = false;
  tensorrt_options->trt_int8_calibration_table_name = "";
  tensorrt_options->trt_int8_use_native_calibration_table = false;
  tensorrt_options->trt_dla_enable = false;
  tensorrt_options->trt_dla_core = 0;
  tensorrt_options->trt_dump_subgraphs = false;
  tensorrt_options->trt_engine_cache_enable = false;
  tensorrt_options->trt_engine_cache_path = "";
  tensorrt_options->trt_engine_decryption_enable = false;
  tensorrt_options->trt_engine_decryption_lib_path = "";
  tensorrt_options->trt_force_sequential_engine_build = false;

  return tensorrt_options;
}
#endif

void padding_resize(cv::InputArray& src, cv::OutputArray& dst, float& fratio,
                    int& ipadd_w, int& ipadd_h, cv::Size size, double fx = 0,
                    double fy = 0, int interpolation = cv::INTER_LINEAR) {
  float padd_w = 0;
  float padd_h = 0;
  // �������ҳ�ԭͼ�б߳�����Ǹ�.Ȼ���Ȱ��ձ߳�����Ǹ�resize,Ȼ��߳�С�ľ����.
  float r =
      std::min(float(size.width) / src.cols(), float(size.height) / src.rows());
  int inside_w = round(src.cols() * r); //
  int inside_h = round(src.rows() * r); //
  padd_w = size.width - inside_w; // padd_w��padd_h����һ������.
  padd_h = size.height - inside_h;
  cout << "padd_w:" << padd_w << ",padd_h:" << padd_h << endl;
  cv::resize(src, dst, cv::Size(inside_w, inside_h), fx, fy, interpolation);
  // padd_w = padd_w / 2;
  // padd_h = padd_h / 2;
  cout << "padd_w:" << padd_w << ",padd_h:" << padd_h << endl;
  //  int top = int(round(padd_h - 0.1));
  int top = 0;
  int bottom = int(round(padd_h + 0.1));
  // int left = int(round(padd_w - 0.1));
  int left = 0;
  int right = int(round(padd_w + 0.1));
  cout << "top:" << top << ",bottom:" << bottom << ",left:" << left
       << ",right:" << right << endl;

  cv::copyMakeBorder(
      dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0));

  fratio = r;
  ipadd_w = int(round(padd_w + 0.1));
  ipadd_h = int(round(padd_h + 0.1));
}

void unpadding_resize(Mat& src, Mat& dst, int ipadd_w, int ipadd_h, int iraw_w,
                      int iraw_h, double fx = 0, double fy = 0,
                      int interpolation = cv::INTER_LINEAR) {
  Mat cropimg;
  cropimg = src(Range(0, src.rows - ipadd_h), Range(0, src.cols - ipadd_w));
  cv::resize(cropimg, dst, cv::Size(iraw_w, iraw_h), fx, fy, interpolation);
}

// norm and bgr to rgb
void PreProcess(const Mat& image, Mat& image_blob) {
  Mat input;
  image.copyTo(input);

  // ���ݴ��� ��׼��
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
}

void run_ort_trt(Mat& img, Mat& mask) {
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test");

#ifdef _WIN32
  const wchar_t* model_path = L"C:/Users/Administrator/Desktop/end2end.onnx";
#else
  const char* model_path =
      "/algdata01/huan.wang/samlabel/playground/mmsegmentation/work_dirs/"
      "ocrnet_hr18_4xb4-80k_ade20k-512x512/end2end.onnx";
#endif

  Ort::Session session(env, model_path, Ort::SessionOptions{nullptr});

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  const size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<int64_t>
      input_node_dims; // simplify... this model has only 1 input node {1, 3,
                       // 224, 224}. Otherwise need vector<vector<>>

  std::cout << "Number of inputs = " << num_input_nodes << std::endl;

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

    // print input node types
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input " << i << " : type = " << type << std::endl;

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    std::cout << "Input " << i << " : num_dims = " << input_node_dims.size()
              << '\n';
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Input " << i << " : dim[" << j
                << "] =" << input_node_dims[j] << '\n';
    }
    std::cout << std::flush;
  }

  // constexpr size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using
  // known dim values to calculate size use OrtGetTensorShapeElementCount() to
  // get official size!

  // std::vector<float> input_tensor_values(input_tensor_size);
  // initialize input data with values in [0.0, 1.0]
  // for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i]
  // = (float)i / (input_tensor_size + 1);

  std::vector<const char*> output_node_names = {"output"};

  clock_t startTime, endTime;

  // input tensor
  startTime = clock();

  // create input tensor object from data values
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // auto input_tensor = Ort::Value::CreateTensor<float>(memory_info,
  // input_tensor_values.data(), input_tensor_size,
  //                                                           input_node_dims.data(),
  //                                                           4);
  std::vector<Ort::Value> input_tensors;

  Mat blob = dnn::blobFromImage(img, 1., Size(img.rows, img.cols),
                                Scalar(0, 0, 0), false, true);

  std::vector<int64_t> input_node_dims_real = {1, 3, img.rows, img.cols};

  input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, blob.ptr<float>(), blob.total(), input_node_dims_real.data(),
      input_node_dims.size()));

  assert(input_tensors[0].IsTensor());

  // score model & input tensor, get back output tensor
  // auto output_tensors =
  //    session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(),
  //    &input_tensor, 1, output_node_names.data(), 1);
  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                  input_tensors.data(), input_node_names.size(),
                  output_node_names.data(), output_node_names.size());
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  endTime = clock();
  std::cout << "The run time is:"
            << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s"
            << std::endl;

  // Get pointer to output tensor float values
  int64_t* intarr = output_tensors.front().GetTensorMutableData<int64_t>();

  mask = Mat_<unsigned char>(img.rows, img.cols); //

  int idx = 0;
  for (int i = 0; i < mask.rows; i++) {
    for (int j = 0; j < mask.cols; j++) {
      mask.at<unsigned char>(i, j) = (unsigned char)(intarr[idx++] * 255);
    }
  }
  std::cout << "Done!" << std::endl;
}

int main1(int argc, char* args) {
  cv::Mat src =
      cv::imread("C:/Users/Administrator/Desktop/road/000014.jpg"); // 720*405
  cv::Mat mask, dst;
  int ipadw, ipadh;
  float fratio;

  // preprocess norm and bgr to rgb
  Mat preprocimg;
  src.convertTo(src, CV_32FC3);
  PreProcess(src, preprocimg);

  // padding and resize
  padding_resize(preprocimg, dst, fratio, ipadw, ipadh, Size(512, 512), 0, 0,
                 cv::INTER_LINEAR);
  imwrite("./padding.jpg", dst);

  run_ort_trt(dst, mask);

  imwrite("./mask.jpg", mask);

  //
  Mat unpadmask;
  unpadding_resize(mask, unpadmask, ipadw, ipadh, src.cols, src.rows, 0, 0,
                   cv::INTER_LINEAR);

  Mat finalmask;
  threshold(unpadmask, finalmask, 128, 255, 0);

  imwrite("./result.jpg", finalmask);

  return 0;
}