#include "func.h"


// 计算旋转不变的LBP特征
Mat computeRotationInvariantLBP(const Mat& src) {
    Mat dst = Mat::zeros(src.size(), CV_8UC1);
    
    for(int i = 1; i < src.rows-1; i++) {
        for(int j = 1; j < src.cols-1; j++) {
            uchar center = src.at<uchar>(i, j);
            unsigned char code = 0;
            
            // 传统的LBP编码
            code |= (src.at<uchar>(i-1, j-1) > center) << 7;
            code |= (src.at<uchar>(i-1, j  ) > center) << 6;
            code |= (src.at<uchar>(i-1, j+1) > center) << 5;
            code |= (src.at<uchar>(i  , j+1) > center) << 4;
            code |= (src.at<uchar>(i+1, j+1) > center) << 3;
            code |= (src.at<uchar>(i+1, j  ) > center) << 2;
            code |= (src.at<uchar>(i+1, j-1) > center) << 1;
            code |= (src.at<uchar>(i  , j-1) > center) << 0;
            
            // 寻找最小LBP值以实现旋转不变
            unsigned char minVal = code;
            unsigned char temp = code;
            for(int k = 0; k < 7; k++) {
                temp = (temp >> 1) | ((temp & 1) << 7);
                if(temp < minVal) {
                    minVal = temp;
                }
            }
            
            dst.at<uchar>(i, j) = minVal;
        }
    }
    
    return dst;
}

// img 图像, templ 模板
// 彩色模板和目标图像会逐通道进行匹配计算，最后将结果合并
// 如果模板与目标图像在颜色空间中有很强的相关性，可能会提高匹配的准确性
Rect detectMatch(cv::Mat &img, cv::Mat &templ, double &bestScore)
{
    bestScore = -1;
    Point bestLoc;
    double bestScale = 1.0;
// 0.5 1.4
    // for (double scale = 0.4; scale <= 2.0; scale += 0.1) {        // 缩放模板
    for (double scale = 0.8; scale <= 1.2; scale += 0.1) {        // 缩放模板

        Mat scaledTempl;
        if(templ.cols<5 || templ.rows<5) continue;//limit
        resize(templ, scaledTempl, Size(), scale, scale);

        // 结果矩阵
        // int result_cols = img.cols - scaledTempl.cols + 1;
        // int result_rows = img.rows - scaledTempl.rows + 1;
        // Mat result(result_rows, result_cols, CV_32FC1);
        Mat result;
        // 模板匹配
        if(img.cols<scaledTempl.cols || img.rows<scaledTempl.rows) continue;//limit
        matchTemplate(img, scaledTempl, result, TM_CCOEFF_NORMED);//归一化相关系数

        double maxVal;
        Point maxLoc;
        minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

        // 更新最佳匹配
        if (maxVal > bestScore) {
            bestScore = maxVal;
            bestLoc = maxLoc;
            bestScale = scale;
        }
    }
#if DEBUG_PRINT
    if(bestScore>0.55) {
        cout << "bestScore: " << bestScore <<endl;
        cout << "bestScale: " << bestScale <<endl;
    }
#endif
    return Rect(bestLoc, Point(bestLoc.x + templ.cols * bestScale, bestLoc.y + templ.rows * bestScale));//min = 6*0.5
}

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <omp.h>

cv::Rect detectMatch_(const cv::Mat& img, const cv::Mat& templ, double& bestScore) {
    bestScore = -1;
    cv::Point bestLoc;
    double bestScale = 1.0;

    // 输入验证
    if (templ.cols < 5 || templ.rows < 5 || img.empty() || templ.empty()) {
        return cv::Rect();
    }

    // 预计算缩放模板
    std::vector<cv::Mat> scaledTemplates;
    std::vector<double> scales = {0.9, 1.0, 1.1};
    for (double scale : scales) {
        cv::Mat scaledTempl;
        cv::resize(templ, scaledTempl, cv::Size(), scale, scale, cv::INTER_LINEAR);
        if (img.cols >= scaledTempl.cols && img.rows >= scaledTempl.rows) {
            scaledTemplates.push_back(scaledTempl);
        }
    }

    if (scaledTemplates.empty()) {
        return cv::Rect();
    }

    // 找到最小模板尺寸以计算最大结果矩阵尺寸
    int minTemplCols = std::min_element(scaledTemplates.begin(), scaledTemplates.end(),
        [](const cv::Mat& a, const cv::Mat& b) { return a.cols < b.cols; })->cols;
    int minTemplRows = std::min_element(scaledTemplates.begin(), scaledTemplates.end(),
        [](const cv::Mat& a, const cv::Mat& b) { return a.rows < b.rows; })->rows;

    // 分配结果矩阵，基于最小模板（最大结果尺寸）
    int maxResultCols = img.cols - minTemplCols + 1;
    int maxResultRows = img.rows - minTemplRows + 1;
    if (maxResultCols <= 0 || maxResultRows <= 0) {
        return cv::Rect(); // 图像太小，无法匹配
    }

    // 设置线程数
    int num_threads = std::min(static_cast<int>(scaledTemplates.size()), omp_get_max_threads());
    omp_set_num_threads(num_threads);

    // 每个线程的局部最佳结果
    struct ThreadResult {
        double maxVal = -1;
        cv::Point maxLoc;
        double scale = 1.0;
    };
    std::vector<ThreadResult> thread_results(num_threads);

    // 并行处理每个尺度
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        cv::Mat result(maxResultRows, maxResultCols, CV_32FC1); // 每个线程独立的 result 矩阵
        ThreadResult local_result; // 线程局部最佳结果

        #pragma omp for schedule(guided, 1) nowait
        for (size_t i = 0; i < scaledTemplates.size(); ++i) {
            const cv::Mat& scaledTempl = scaledTemplates[i];
            double scale = scales[i];

            // 计算当前模板的结果尺寸
            int result_cols = img.cols - scaledTempl.cols + 1;
            int result_rows = img.rows - scaledTempl.rows + 1;

            // 确保结果尺寸有效
            if (result_cols <= 0 || result_rows <= 0 || 
                result_cols > maxResultCols || result_rows > maxResultRows) {
                continue;
            }

            // 创建结果视图
            cv::Mat resultView = result(cv::Rect(0, 0, result_cols, result_rows));


            // cv::Mat imgLBP = computeRotationInvariantLBP(img);
            // cv::Mat scaledTemplLBP = computeRotationInvariantLBP(scaledTempl);
            // cv::matchTemplate(imgLBP, scaledTemplLBP, resultView, cv::TM_CCOEFF_NORMED);

            // 模板匹配
            cv::matchTemplate(img, scaledTempl, resultView, cv::TM_CCOEFF_NORMED);

            double maxVal;
            cv::Point maxLoc;
            cv::minMaxLoc(resultView, nullptr, &maxVal, nullptr, &maxLoc);

            // 更新线程局部最佳匹配
            if (maxVal > local_result.maxVal) {
                local_result.maxVal = maxVal;
                local_result.maxLoc = maxLoc;
                local_result.scale = scale;
            }
        }

        // 保存线程局部结果
        thread_results[thread_id] = local_result;
    }

    // 汇总所有线程的最佳匹配
    for (const auto& tr : thread_results) {
        if (tr.maxVal > bestScore) {
            bestScore = tr.maxVal;
            bestLoc = tr.maxLoc;
            bestScale = tr.scale;
        }
    }
    std::cout << "bestScore = " << bestScore << std::endl;
    // 使用整数运算计算最终矩形
    int width = static_cast<int>(templ.cols * bestScale);
    int height = static_cast<int>(templ.rows * bestScale);
    return cv::Rect(bestLoc, cv::Point(bestLoc.x + width, bestLoc.y + height));
}


void getOptimalROISize(cv::Rect &bbox)
{
    int new_width = cv::getOptimalDFTSize(bbox.width); // 自动调整为 2 的幂
    int new_height = cv::getOptimalDFTSize(bbox.height); // 自动调整为 2 的幂

    bbox.x -= (new_width - bbox.width)/2;
    bbox.y -= (new_height - bbox.height)/2;
    bbox.height = new_height;
    bbox.width = new_width;
}



// Point 限幅函数
cv::Point clampPoint(const cv::Point& pt, const cv::Mat& img) {
    // 确保 x 和 y 坐标在图像范围内
    int x = std::max(0, std::min(pt.x, img.cols - 1));
    int y = std::max(0, std::min(pt.y, img.rows - 1));
    return cv::Point(x, y);
    // return cv::Point(std::clamp(pt.x, 0, img.cols - 1), std::clamp(pt.y, 0, img.rows - 1));
}

// Rect 限幅函数 输入中心点
// cv::Rect clampROI(const cv::Point& pt, const cv::Size& roiSize_, const cv::Mat& img) {
//     cv::Point pc = clampPoint(pt, img);
//     int x1 = max(pc.x - roiSize_.width/2, 0);
//     int y1 = max(pc.y - roiSize_.height/2, 0);
//     int x2 = min(pc.x + roiSize_.width/2, img.cols - 1);
//     int y2 = min(pc.y + roiSize_.height/2, img.rows - 1);
//     return cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
// }


cv::Rect clampROI(const cv::Point& pt, const cv::Size& roiSize_, const cv::Mat& img) {
    cv::Point pc = clampPoint(pt, img);
    int x1 = pc.x - roiSize_.width/2;
    int y1 = pc.y - roiSize_.height/2;
    return (Rect(x1, y1, roiSize_.width, roiSize_.height) & Rect(0, 0, img.cols, img.rows));
}


// Rect 限幅函数
// cv::Rect clampRect(Rect& rect_, const Mat& img) {
//     int x = min((rect_.x + rect_.width), img.cols - 1);
//     int y = min((rect_.y + rect_.height), img.rows - 1);
//     return Rect(rect_.tl(), Point(x, y));
// }

// Rect 限幅函数
cv::Rect clampRect(Rect& rect_, const Mat& img) {
    int x1 = std::max(rect_.x, 0);
    int y1 = std::max(rect_.y, 0);
    int x2 = min(rect_.x + rect_.width, img.cols - 1);
    int y2 = min(rect_.y + rect_.height, img.rows - 1);
    // 避免负宽高
    if (x2 < x1 || y2 < y1) {
        return cv::Rect(0, 0, 0, 0);  // 返回空矩形
    }
    return Rect(Point(x1, y1), Point(x2, y2));
}

// 按文件名顺序获取图片路径
vector<string> getImageSequence(const string& folderPath, const string& extension) {
    vector<string> fileNames;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == extension) {
            fileNames.push_back(entry.path().string());
        }
    }
    sort(fileNames.begin(), fileNames.end()); // 按文件名排序
    return fileNames;
}

// 调整画面尺寸为原尺寸的一半
void showScaleImg(std::string str, const cv::Mat &frame)
{
    #ifndef SCALE
    #define SCALE 2
    #endif
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(frame.cols / SCALE, frame.rows / SCALE));
    // 显示结果
    cv::imshow(str, resizedFrame);
}

// 鼠标回调函数
void mouseCallback(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) { // 左键按下，开始绘制
        drawing = true;
        startPoint = Point(x * SCALE, y * SCALE);
    } else if (event == EVENT_MOUSEMOVE && drawing) { // 鼠标移动，更新目标框
        endPoint = Point(x * SCALE, y * SCALE);
    } else if (event == EVENT_LBUTTONUP) { // 左键抬起，结束绘制
        drawing = false;
        endPoint = Point(x * SCALE, y * SCALE);
        bbox = Rect(startPoint, endPoint);
        cout << bbox << endl;
    } else if(event == EVENT_MBUTTONDOWN) {
        startPoint = Point(x * SCALE, y * SCALE);
        bbox = Rect(startPoint.x - ROI_SIZE/2, startPoint.y - ROI_SIZE/2, ROI_SIZE, ROI_SIZE);
        cout << bbox << endl;
    }
}

cv::Rect computeBox(const cv::Mat& image, const cv::Rect& bbox) {
    int xExpand = bbox.width/6;//4 //6 //8
    int yExpand = bbox.height/6;
    cv::Rect bboxExpand(cv::Point(bbox.x - xExpand, bbox.y - yExpand), cv::Point(bbox.br().x + xExpand, bbox.br().y + yExpand));
    bboxExpand = clampRect(bboxExpand, image);// limit
    cv::Mat imgRoiExpand = image(bboxExpand);// cal a part
#if DEBUG_PRINT
    //rectangle(tempFrame, bboxExpand, cv::Scalar(200, 50, 100), 1, 8, 0);
    //cv::imshow("imgRoiExpand", imgRoiExpand);
#endif
    cv::Rect bboxRect(bbox.x - bboxExpand.x, bbox.y - bboxExpand.y, bbox.width, bbox.height);// offset

    // 初始化掩膜和模型./ed 
    cv::Mat mask = cv::Mat::zeros(imgRoiExpand.size(), CV_8UC1);
    cv::Mat bgModel, fgModel;
    // 执行 GrabCut
    cv::grabCut(imgRoiExpand, mask, bboxRect, bgModel, fgModel, 5, cv::GC_INIT_WITH_RECT);
    // 创建前景掩膜
    cv::compare(mask, cv::GC_PR_FGD, mask, cv::CMP_EQ);
#if DEBUG_PRINT
    // // cv::imshow("mask", mask);
    // // cv::Mat segmented_image = cv::Mat::zeros(imgRoiExpand.size(), imgRoiExpand.type());
    // // imgRoiExpand.copyTo(segmented_image, mask);
    // // cv::imshow("segmented_image", segmented_image);
#endif
    // 轮廓提取
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 查找最大面积轮廓
    double maxArea = 0;
    std::vector<cv::Point> maxContour;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
            maxContour = contour;
        }
    }
    // 提取最大轮廓外接矩形
    if (!maxContour.empty()) {
        cv::Rect boundingBox = cv::boundingRect(maxContour);
        boundingBox.x = boundingBox.x + bboxExpand.x;
        boundingBox.y = boundingBox.y + bboxExpand.y;
        return boundingBox;
    }
    return bbox;
}


#define DEBUG_PRINT_ 1
#define ORB_M 1
// img1模板 img2搜索区域
cv::Rect detectAndMatch(cv::Mat& img1, cv::Mat& img2, double &bestScore) {
#if ORB_M
    // 1. 初始化 ORB 特征检测器
    cv::Ptr<cv::ORB> sift = cv::ORB::create();
#else
    #if 0
    // 1. 初始化 SIFT 检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
#else
    cv::Ptr<cv::BRISK> sift = cv::BRISK::create();
#endif
#endif
    // 2. 检测关键点和描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
    if(descriptors1.empty() || descriptors2.empty()) {
        cout << "descriptors1 empty..." << endl;
        return Rect{0, 0, img2.cols, img2.rows};
    }
#if 1
    #if ORB_M
        cv::Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_HAMMING); // cv::NORM_L2
    #else
        cv::Ptr<BFMatcher> matcher = BFMatcher::create(cv::NORM_L2); // cv::
    #endif
#else
    cv::Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
#endif

    // 筛选匹配点：双向匹配和比值测试
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);
    std::vector<cv::DMatch> goodMatches;
    for (const auto& knnMatch : knnMatches) {
        if (knnMatch[0].distance < 0.7 * knnMatch[1].distance) {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    // 5. 计算单应性矩阵
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : goodMatches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    if(points1.size() < 4 || points2.size() < 4) {
        cout << "points1.size() < 4 " << endl;
        return Rect{0, 0, img2.cols, img2.rows};
    }

#if 0 //DEBUG_PRINT
    // 7. 绘制匹配点和位置
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);
    cv::imshow("Matches & Object Location", imgMatches);
    // cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);
    waitKey(0);
#endif
    Mat inliersMask;
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 3, inliersMask);
    // 统计内点数量
    int inliersCount = countNonZero(inliersMask);
    cout << "Total matches: " << goodMatches.size() << endl;
    cout << "Inliers count: " << inliersCount << endl;
    cout << "points1 matches: " << points1.size() << endl;
    cout << "keypoints1 matches: " << keypoints1.size() << endl;

    float score = 0;
    // 判断目标是否存在
    if (inliersCount > 8) {
        cout << "Target detected in search region!" << endl;
        score = inliersCount / (keypoints1.size() * 0.7);
        if(score>1) score = 1.0;
        // // 可视化：绘制内点匹配
        // vector<DMatch> inlierMatches;
        // for (size_t i = 0; i < goodMatches.size(); i++) {
        //     if (inliersMask.at<uchar>(i)) {
        //         inlierMatches.push_back(goodMatches[i]);
        //     }
        // }
        // Mat inliersimgMatches;
        // cv::drawMatches(img1, keypoints1, img2, keypoints2, inlierMatches, inliersimgMatches);
        // cout << "inlierMatches" << inlierMatches.size() << endl;
        // imshow("Inlier Matches", inliersimgMatches);
    } else {
        cout << "Target not found or insufficient inliers!" << endl;
        score = 0;
        return Rect{0, 0, img2.cols, img2.rows};
    }

    // 6. 计算目标在大图像中的位置
    std::vector<cv::Point2f> objCorners(4);
    objCorners[0] = cv::Point2f(0, 0);
    objCorners[1] = cv::Point2f((float)img1.cols, 0);
    objCorners[2] = cv::Point2f((float)img1.cols, (float)img1.rows);
    objCorners[3] = cv::Point2f(0, (float)img1.rows);
    std::vector<cv::Point2f> sceneCorners(4);
    cv::perspectiveTransform(objCorners, sceneCorners, H);

#if DEBUG_PRINT_
    // 7. 绘制匹配点和位置
    cv::Mat imgMatches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);

    // 在大图像上绘制矩形框
    cv::line(imgMatches, sceneCorners[0] + cv::Point2f((float)img1.cols, 0), sceneCorners[1] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4);
    cv::line(imgMatches, sceneCorners[1] + cv::Point2f((float)img1.cols, 0), sceneCorners[2] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4);
    cv::line(imgMatches, sceneCorners[2] + cv::Point2f((float)img1.cols, 0), sceneCorners[3] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4);
    cv::line(imgMatches, sceneCorners[3] + cv::Point2f((float)img1.cols, 0), sceneCorners[0] + cv::Point2f((float)img1.cols, 0), cv::Scalar(0, 255, 0), 4);

    cout << "sceneCorners[0]: " << sceneCorners[0] <<endl;
    cout << "sceneCorners[1]: " << sceneCorners[1] <<endl;
    cout << "sceneCorners[2]: " << sceneCorners[2] <<endl;
    cout << "sceneCorners[3]: " << sceneCorners[3] <<endl;
#endif

    cv::Point m_center = (sceneCorners[0] + sceneCorners[1] + sceneCorners[2] + sceneCorners[3]) * 0.25;
    int m_width = (sceneCorners[1] + sceneCorners[2] - sceneCorners[0] - sceneCorners[3]).x * 0.5;
    int m_height = (sceneCorners[2] + sceneCorners[3] - sceneCorners[0] - sceneCorners[1]).y * 0.5;
    cv::Rect m_rect(m_center.x - m_width/2, m_center.y - m_height/2, m_width, m_height);

    if(score>0.32) bestScore = 1;
#if DEBUG_PRINT_
    cout << "score: " << score << endl;
    cout << "m_center: " << m_center << endl;
    cout << "m_rect: " << m_rect << endl;
    cout << "bestScore: " << bestScore << endl;

    
    cv::Rect showRect(m_rect.x + img1.cols, m_rect.y + 0, m_rect.width, m_rect.height);
    cv::rectangle(imgMatches, showRect, cv::Scalar(255, 0, 255), 2);
    cv::circle(imgMatches, m_center + cv::Point(img1.cols, 0), 5, cv::Scalar(255, 0, 255), -1);
    // 显示结果
    cv::imshow("Matches & Object Location", imgMatches);
    imwrite("file1.jpg", imgMatches);
#endif
    return m_rect;
}


cv::Ptr<cv::ml::SVM> trainSVM(const std::vector<cv::Mat>& samples, const std::vector<int>& labels) {
    cv::Mat trainData, trainLabels;
    
    for (size_t i = 0; i < samples.size(); ++i) {
        cv::Mat sampleRow;
        if(samples[i].isContinuous()) {
            sampleRow = samples[i].reshape(1, 1); // 展平
            std::cout << "isContinuous" << std::endl;
        } else {
            sampleRow = samples[i].clone().reshape(1, 1); // 展平
            std::cout << "not Continuous" << std::endl;
        }
        trainData.push_back(sampleRow);
        trainLabels.push_back(labels[i]);
    }

    trainData.convertTo(trainData, CV_32F);
    trainLabels.convertTo(trainLabels, CV_32S);

    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(trainData, cv::ml::ROW_SAMPLE, trainLabels);

    return svm;
}

cv::Rect reDetect(const cv::Mat& image, cv::Ptr<cv::ml::SVM> svm, cv::Size windowSize, double &maxScore) {
    maxScore = -DBL_MAX;
    cv::Rect bestRect;

    for (int y = 0; y <= image.rows - windowSize.height; y += 4) {
        for (int x = 0; x <= image.cols - windowSize.width; x += 4) {
            cv::Rect roi(x, y, windowSize.width, windowSize.height);
            cv::Mat patch = image(roi).clone().reshape(1, 1);
            patch.convertTo(patch, CV_32F);

            float score = svm->predict(patch, cv::noArray(), cv::ml::StatModel::RAW_OUTPUT);
            if (score > maxScore) {
                maxScore = score;
                bestRect = roi;
            }
        }
    }
    std::cout << "maxScore = " << maxScore << std::endl;
    return bestRect;
}

Mat featureExtractor(const Mat& roi) {
    HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
    vector<float> descriptors;
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    resize(gray, gray, Size(64, 128));
    hog.compute(gray, descriptors);

    return Mat(descriptors).reshape(1, 1); // 1行N列
}


#include <arm_neon.h> // ARM NEON 头文件

// 计算单点的 NCC 值（NEON 加速）
float computeNCC_NEON(const uint8_t* templ, const uint8_t* img_patch, int templ_size) {
    uint32x4_t sum_ti = vdupq_n_u32(0); // T*I 的累加器
    uint32x4_t sum_t2 = vdupq_n_u32(0); // T^2 的累加器
    uint32x4_t sum_i2 = vdupq_n_u32(0); // I^2 的累加器

    for (int i = 0; i < templ_size; i += 16) {
        // 加载 16 个像素（128-bit 寄存器）
        uint8x16_t v_templ = vld1q_u8(templ + i);
        uint8x16_t v_img = vld1q_u8(img_patch + i);

        // 计算 T*I（8x8 -> 16-bit，再累加到 32-bit）
        uint16x8_t ti_low = vmull_u8(vget_low_u8(v_templ), vget_low_u8(v_img));
        uint16x8_t ti_high = vmull_u8(vget_high_u8(v_templ), vget_high_u8(v_img));
        sum_ti = vaddq_u32(sum_ti, vpaddlq_u16(ti_low));
        sum_ti = vaddq_u32(sum_ti, vpaddlq_u16(ti_high));

        // 计算 T^2 和 I^2（同理）
        uint16x8_t t2_low = vmull_u8(vget_low_u8(v_templ), vget_low_u8(v_templ));
        uint16x8_t t2_high = vmull_u8(vget_high_u8(v_templ), vget_high_u8(v_templ));
        sum_t2 = vaddq_u32(sum_t2, vpaddlq_u16(t2_low));
        sum_t2 = vaddq_u32(sum_t2, vpaddlq_u16(t2_high));

        uint16x8_t i2_low = vmull_u8(vget_low_u8(v_img), vget_low_u8(v_img));
        uint16x8_t i2_high = vmull_u8(vget_high_u8(v_img), vget_high_u8(v_img));
        sum_i2 = vaddq_u32(sum_i2, vpaddlq_u16(i2_low));
        sum_i2 = vaddq_u32(sum_i2, vpaddlq_u16(i2_high));
    }

    // 提取累加结果
    float s_ti = vgetq_lane_u32(sum_ti, 0) + vgetq_lane_u32(sum_ti, 1) + 
                 vgetq_lane_u32(sum_ti, 2) + vgetq_lane_u32(sum_ti, 3);
    float s_t2 = vgetq_lane_u32(sum_t2, 0) + vgetq_lane_u32(sum_t2, 1) + 
                 vgetq_lane_u32(sum_t2, 2) + vgetq_lane_u32(sum_t2, 3);
    float s_i2 = vgetq_lane_u32(sum_i2, 0) + vgetq_lane_u32(sum_i2, 1) + 
                 vgetq_lane_u32(sum_i2, 2) + vgetq_lane_u32(sum_i2, 3);

    // 计算 NCC
    return s_ti / (sqrt(s_t2 * s_i2) + 1e-6); // 避免除零
}