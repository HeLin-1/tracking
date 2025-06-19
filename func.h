#ifndef __FUNC_H__
#define __FUNC_H__

#define CV_KCF 0
#define ROI_RESIZE 0
#define GRAY_INPUT 1

#include "kcftracker.hpp"
#include "fdssttracker.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#if CV_KCF
    #include <opencv2/tracking.hpp>
#endif
#include <opencv2/core/utility.hpp>
#include <iostream>
#include <filesystem>
#include <queue>


using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define SCALE 1.5
#define IMG_SEQ 0

#define DEBUG_PRINT 1
#define DATA 0

#define ROI_SIZE 64

// 全局变量
extern bool drawing;                // 是否正在绘制
extern Rect bbox;                   // 目标框
extern Point startPoint, endPoint;  // 起点和终点

// 按文件名顺序获取图片路径
vector<string> getImageSequence(const string& folderPath, const string& extension);
// 调整画面尺寸为原尺寸的一半
void showScaleImg(string str, const cv::Mat &frame);
// 限幅函数，参数改为 Mat
cv::Point clampPoint(const Point& pt, const Mat& img);
cv::Rect clampROI(const Point& pt, const Size& roiSize_, const Mat& img);
// Rect 限幅函数
cv::Rect clampRect(Rect& rect_, const Mat& img);
// img 图像, templ 模板
// 彩色模板和目标图像会逐通道进行匹配计算，最后将结果合并
// 如果模板与目标图像在颜色空间中有很强的相关性，可能会提高匹配的准确性
Rect detectMatch(cv::Mat &img, cv::Mat &templ, double &bestScore);
cv::Rect detectMatch_(const cv::Mat& img, const cv::Mat& templ, double& bestScore);

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int, void*);
// edgebox 缩小边框
cv::Rect computeBox(const cv::Mat& image, const cv::Rect& bbox);
// say hello
void say_hello();

// 优化ROI尺寸和位置
void getOptimalROISize(cv::Rect &bbox);

// img1模板 img2搜索区域
cv::Rect detectAndMatch(cv::Mat& img1, cv::Mat& img2, double &bestScore);

cv::Ptr<cv::ml::SVM> trainSVM(const std::vector<cv::Mat>& samples, const std::vector<int>& labels);

// cv::Rect reDetect(const cv::Mat& image, cv::Ptr<cv::ml::SVM> svm, cv::Size windowSize);
cv::Rect reDetect(const cv::Mat& image, cv::Ptr<cv::ml::SVM> svm, cv::Size windowSize, double &maxScore);



#endif
