/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/


#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "kcftracker.hpp"
#include "fdssttracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<cv::Rect> GetGroundtruth(std::string txt_file);
std::vector<double>PrecisionCalculate(std::vector<cv::Rect> groundtruth_rect, 
				      std::vector<cv::Rect> result_rect);
// make -j4 && ./test /home/lin/Downloads/Human6 1
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage:" 
              << argv[0] << " video_base_path[./David3] Verbose[0/1]" << std::endl;
    return 0;
  }
  std::string video_base_path = argv[1];
  std::string pattern_jpg = video_base_path+ "/img/*.jpg";
  std::string pattern_png = video_base_path+ "/img/*.png";
  std::vector<cv::String> image_files;
  cv::glob(pattern_jpg, image_files);
  if (image_files.size() == 0)
    cv::glob(pattern_png, image_files);
  if (image_files.size() == 0) {
    std::cout << "No image files[jpg png]" << std::endl;
    return 0;
  }
  std::string txt_base_path = video_base_path + "/groundtruth_rect.txt"; // groundtruth_rect groundtruth
  std::vector<cv::Rect> groundtruth_rect;
  groundtruth_rect = GetGroundtruth(txt_base_path);

  cv::Mat image;
  cv::Mat imageGray;
  std::vector<cv::Rect> result_rect;
  int64 tic, toc;
  double time = 0;
  bool show_visualization = argc == 3? atoi(argv[2]) : true;

  bool track_sucess = false;
  uint16_t track_sucess_cnt = 1;

    bool LAB = false;
    bool HOG = true;
    bool FIXEDWINDOW = true;
#define FDSST
#ifndef FDSST
    bool MULTISCALE = true; // true
    KCFTracker *tracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
#else
    bool MULTISCALE = true;
    FDSSTTracker *tracker = new FDSSTTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
#endif

  for(unsigned int frame = 0; frame < image_files.size(); ++frame) {
    image = cv::imread(image_files[frame]);
    if(image.channels() == 3)
        cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    // std::cout << "image.channels() = " << image.channels() << std::endl;
    tic = getTickCount();
    if (frame == 0) {
        tracker->init(groundtruth_rect[0], imageGray);
        result_rect.push_back(groundtruth_rect[0]); //0-index
    } else {

    Rect bbox;
    #ifndef FDSST
      track_sucess = tracker->update(image, bbox);
    #else
      track_sucess = tracker->update(imageGray, bbox);
    #endif
    result_rect.push_back(bbox);

    // result_rect.push_back(tracker->update(imageGray));
    }
    toc = cv::getTickCount() - tic;
    if(track_sucess) // 跟踪成功，计算时间 // 
    {
      track_sucess_cnt++;
      time += toc;
    }

    if (show_visualization) {
      cv::putText(image, to_string(frame+1), cv::Point(20, 40), 6, 1,
                  cv::Scalar(0, 255, 255), 2);
      cv::rectangle(image, groundtruth_rect[frame], cv::Scalar(0, 255, 0), 1);
      cv::rectangle(image, result_rect[frame], cv::Scalar(255, 0, 255), 2);
      cv::imshow(video_base_path, image);
    
      char key = cv::waitKey(2); // 2
      if(key == 27 || key == 'q' || key == 'Q')
        break;
    }
  }
  time = time / double(getTickFrequency());
  // double fps = double(result_rect.size()) / time;
  double fps = double(track_sucess_cnt) / time;

  
  std::vector<double> precisions = PrecisionCalculate(groundtruth_rect,
                                                      result_rect);
  printf("%12s - Precision (20px) : %1.3f, FPS : %4.2f\n",
         video_base_path.c_str(), precisions[20], fps);

  cout << "track_sucess_cnt = " << track_sucess_cnt << endl;
  cout << "result_rect.size = " << result_rect.size() << endl;

  
  cv::destroyAllWindows();

  return 0;
}


std::vector<cv::Rect> GetGroundtruth(std::string txt_file) {
  std::vector<cv::Rect> rect;
  ifstream gt;
  gt.open(txt_file.c_str());
  if (!gt.is_open())
    std::cout << "Ground truth file " << txt_file 
              << " can not be read" << std::endl;
  std::string line;
  int tmp1, tmp2, tmp3, tmp4;
  while (getline( gt, line)) {
    std::replace(line.begin(), line.end(), ',', ' ');
    stringstream ss;
    ss.str(line);
    ss >> tmp1 >> tmp2 >> tmp3 >> tmp4;
    rect.push_back( cv::Rect(--tmp1, --tmp2, tmp3, tmp4) ); //0-index
  }
  gt.close();
  return rect;
}


std::vector<double> PrecisionCalculate(std::vector<cv::Rect>groundtruth_rect,
                                       std::vector<cv::Rect>result_rect) {
  int max_threshold = 50;
  std::vector<double> precisions(max_threshold + 1, 0);
  if (groundtruth_rect.size() != result_rect.size()) {
    int n = min(groundtruth_rect.size(), result_rect.size());
    groundtruth_rect.erase(groundtruth_rect.begin()+n, groundtruth_rect.end());
    result_rect.erase(result_rect.begin() + n, result_rect.end());
  }
  std::vector<double> distances;
  double distemp;
  for (unsigned int i = 0; i < result_rect.size(); ++i) {
    distemp = sqrt(double(pow((result_rect[i].x + result_rect[i].width / 2) -
              (groundtruth_rect[i].x + groundtruth_rect[i].width / 2), 2) +
                          pow((result_rect[i].y + result_rect[i].height / 2) -
              (groundtruth_rect[i].y + groundtruth_rect[i].height / 2), 2)));
    distances.push_back(distemp);
  }
  for (int i = 0; i <= max_threshold; ++i) {
    for (unsigned int j = 0; j < distances.size(); ++j) {
      if (distances[j] < double(i))
        precisions[i]++;
    }
    precisions[i] = precisions[i] / distances.size();
  }
  return precisions;
}