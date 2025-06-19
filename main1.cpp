#include "func.h"

// #include <opencv2/video/tracking.hpp>
#include "KalmanTracker.h"

#include <opencv2/tracking.hpp>

#include "BYTETracker.h" //bytetrack



void test_bytetrack(cv::Mat& frame, std::vector<detect_result>& results,BYTETracker& tracker)
{
    std::vector<detect_result> objects;


    for (detect_result dr : results) {

        if(dr.classId == 1) {
            objects.push_back(dr);
        }
    }

    std::vector<STrack> output_stracks = tracker.update(objects);

    for (unsigned long i = 0; i < output_stracks.size(); i++) {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        // std::cout << "tlwh[0], tlwh[1], tlwh[2], tlwh[3] : " << tlwh[0] << ", " << tlwh[1] << ", " << tlwh[2] << ", " << tlwh[3] << ", " << std::endl; 

        // bool vertical = tlwh[2] / tlwh[3] > 1.6; // 宽 / 高
        bool vertical = true;
        if (tlwh[2] * tlwh[3] > 20 && vertical) {
            cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
            cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                    0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);

            // std::cout << "cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]) : " << cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]) << std::endl; 
        }
    }
}
//bytetrack
int fps = 30; // 20
BYTETracker bytetracker(fps, 60);
std::vector<detect_result> results{{0, 0.0f, cv::Rect_<float>(0, 0, 0, 0)}};


#include <sys/select.h>
#include <time.h>

void nonblocking_delay_us(long microseconds) {
    struct timeval tv;
    tv.tv_sec = microseconds / 1000000;
    tv.tv_usec = microseconds % 1000000;
    select(0, NULL, NULL, NULL, &tv);
}

// 全局变量
bool drawing = false;       // 是否正在绘制
Rect bbox;                  // 目标框
Rect bbox_csrt;                  // 目标框
Point startPoint, endPoint; // 起点和终点
Mat tempFrame;              // 临时帧用于实时绘制
Mat initRecImg;

cv::TickMeter time_ms;
cv::TickMeter rec_time_ms;
//借用 

bool m_retest = false;
uint32_t m_retest_cnt = 0;
uint32_t m_match_cnt = 0;
#if DATA
    uint32_t m_frameStartId = 1700;//0 car 5-4100-1700//car-4 0
#else
    uint32_t m_frameStartId = 550;//0 car 5-4100-1700//car-4 0
#endif



cv::Mat frame_sub;
cv::Size targetSize(640, 360); // 目标尺寸 (宽 300, 高 200)
cv::Rect bbox_sub;

#include <deque>
#include <algorithm>

// 定义存储图像和 m_apceValue 的结构体
struct FrameData {
    cv::Mat image;       // 图像
    float m_apceValue;   // 对应的数值

    // 构造函数
    FrameData(const cv::Mat& img, float value) : image(img.clone()), m_apceValue(value) {}
};

// 用于排序的比较函数
bool compareFrameData(const FrameData& a, const FrameData& b) {
    return a.m_apceValue < b.m_apceValue; // 升序排序
}
static int re_detect_cnt = 0;


// 画角框的函数
void drawCornerBox(cv::Mat& img, const cv::Rect& box, int thickness = 2, const cv::Scalar& color = {0, 255, 0}, int line_len = 20) {
    int x = box.x;
    int y = box.y;
    int w = box.width;
    int h = box.height;

    line_len = w / 6 > 20 ? 20 : w / 6;
    // 左上
    cv::line(img, {x, y}, {x + line_len, y}, color, thickness);
    cv::line(img, {x, y}, {x, y + line_len}, color, thickness);

    // 右上
    cv::line(img, {x + w, y}, {x + w - line_len, y}, color, thickness);
    cv::line(img, {x + w, y}, {x + w, y + line_len}, color, thickness);

    // 左下
    cv::line(img, {x, y + h}, {x + line_len, y + h}, color, thickness);
    cv::line(img, {x, y + h}, {x, y + h - line_len}, color, thickness);

    // 右下
    cv::line(img, {x + w, y + h}, {x + w - line_len, y + h}, color, thickness);
    cv::line(img, {x + w, y + h}, {x + w, y + h - line_len}, color, thickness);
}

int main(int argc, char **arg) {
#if !IMG_SEQ
    // 打开视频文件或摄像头
    // string videoPath = "/home/lin/Downloads/GOT-10k_val_partial/GOT-10k_Val_000006.mp4"; // 替换为视频路径或设置为 "0" 使用摄像头
    string videoPath = "/home/lin/Downloads/";
    videoPath += "vlc-record-2025-04-08-16h41m15s-rtsp___192.168.1.22_chn0-.mp4";
    // videoPath +="mmexport1749189379718.mp4";

    // videoPath += "GOT-10k_val_partial/GOT-10k_Val_000006.mp4";
    // videoPath = "/media/lin/bootfs/vlc-record-2025-04-02-09h20m53s-rtsp___192.168.1.22_chn0-.mp4";
    // videoPath = "/media/lin/bootfs/vlc-record-2025-04-02-09h27m18s-rtsp___192.168.1.22_chn0-.mp4";

    // videoPath = arg[2];
    
    VideoCapture cap(videoPath);


    // 检查是否成功打开视频
    if (!cap.isOpened()) {
        cout << "无法打开视频或摄像头" << endl;
        return -1;
    }

    // 读取第一帧
    Mat frame;
    if (!cap.read(frame)) {
        cout << "无法读取视频帧" << endl;
        return -1;
    }
#if GRAY_INPUT
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
#endif
#else

#define ENABLE_CROSS_X 1
#ifdef ENABLE_CROSS_X
    string folderPath = "/home/lin/Downloads/"; //car-18 car-5
#else
    string folderPath = "/mnt/d/Dataset/"; //car-18 car-5
#endif
    // 图片序列文件夹路径
    #if DATA
        // string folderPath = "/mnt/d/Dataset/car/car-5/img"; //car-18 car-5
        folderPath += "car/car-5/img"; 
    #else
        //string folderPath = "/home/lin/Downloads/car/car-4/img";
        //string folderPath = "/home/lin/Downloads/car/car-20/img";
        //276*120--444,354
        //// string folderPath = "/home/lin/Downloads/car/car-11/img";
        //string folderPath = "/home/lin/Downloads/car/car-18/img";
        // string folderPath = "/mnt/d/Dataset/car/car-18/img"; //car-18 car-5
        // string folderPath = "/mnt/d/Dataset/car/car-4/img"; //car-18 car-5

        // folderPath += "car/car-4/img"; // 遮挡，无旋转
        folderPath += "car/car-18/img";
    #endif
    string extension = ".jpg";// 图片格式
    
    // 获取图片序列
    vector<string> imageFiles = getImageSequence(folderPath, extension);
    if (imageFiles.empty()) {
        cout << "未找到图片序列！请检查路径或扩展名。" << endl;
        return -1;
    }
     // 读取第一帧并选择目标框
#if GRAY_INPUT
    Mat frame = imread(imageFiles[m_frameStartId], cv::IMREAD_GRAYSCALE);// gray
    // cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
#else
    Mat frame = imread(imageFiles[m_frameStartId]);// color
#endif

    std::cout << "frame.size: " << frame.size() << std::endl;
    cout << "imageFiles.size: " <<imageFiles.size() << endl;

#endif
    // 设置窗口并绑定鼠标回调函数
    namedWindow(u8"选择目标框", WINDOW_AUTOSIZE);
    tempFrame = frame.clone();
    if(tempFrame.channels() == 1) {
        cv::cvtColor(tempFrame, tempFrame, cv::COLOR_GRAY2BGR);
    }
    //imwrite("./d.jpg", tempFrame);
    setMouseCallback(u8"选择目标框", mouseCallback);

#if 1

    // 等待用户按下 Enter 键确认目标框
    while (true) {
        Mat tempFrame1 = tempFrame.clone(); // 还原帧
        //rectangle(tempFrame1, startPoint, endPoint, Scalar(0, 255, 0), 2);
        rectangle(tempFrame1, bbox, Scalar(0, 255, 0), 2);

        showScaleImg(u8"选择目标框", tempFrame1);

        int key = waitKey(0); // 按下了键盘上的任何一个键，waitKey() 函数会立即返回所按键的 ASCII 码，并且等待时间会提前结束
        if (key == 13) { // Enter 键
            break; // 跳出循环
        } else if(key == 27) { // Esc
            return 0; // 退出程序
        }

        // 动态更新
#if IMG_SEQ
        frame = imread(imageFiles[++m_frameStartId], cv::IMREAD_GRAYSCALE);
#else
        cap.read(frame);
#endif
        if(frame.channels() == 3)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        tempFrame = frame.clone();
        if(tempFrame.channels() == 1) {
            cv::cvtColor(tempFrame, tempFrame, cv::COLOR_GRAY2BGR);
        }
        if (bbox.width) { // 中键
            cout << "中键2" << endl;
            break;
        }
        // 动态更新
    }
    destroyWindow(u8"选择目标框");
#else
    // bbox = Rect(Point(446, 364), Point(446+256, 364+96));
    bbox = Rect(Point(442, 360), Point(442+224, 360+106));
    // bbox = Rect(Point(438, 358), Point(438+234, 358+102));
#endif

    // roi 区域目标分割
    cv::Rect boundingBox = computeBox(tempFrame, bbox);
    // 绘制外接矩形
    // cv::rectangle(tempFrame, boundingBox, cv::Scalar(0, 0, 255), 1);
    std::cout << "Bounding Box of Largest Contour: " << boundingBox << std::endl; // 输出外接矩形坐标
    cv::rectangle(tempFrame, bbox, cv::Scalar(0, 255, 0), 1);

    // bbox = boundingBox; // adjust

    std::cout << "bbox.tl(): " << bbox.tl() << std::endl;
    std::cout << "bbox.size(): " << bbox.size() << std::endl;

    // getOptimalROISize(bbox);

    std::cout << "Optimal bbox.tl(): " << bbox.tl() << std::endl;
    std::cout << "Optimal bbox.size(): " << bbox.size() << std::endl;

    cv::rectangle(tempFrame, bbox, cv::Scalar(255, 0, 0));
    showScaleImg("boundingBox", tempFrame);


    // 扩展图像到最佳尺寸
    // cv::Mat padded;
    // cv::copyMakeBorder(image, padded, 0, optimal_rows - image.rows,
    // 0, optimal_cols - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    /*
    if (bbox.width == 0 || bbox.height == 0) {
        cout << "未选择有效的目标框" << endl;
        return -1;
    } else {
        initRecImg = tempFrame(bbox);
        imshow("initRecImg", initRecImg);
        //imwrite("./c.jpg", initRecImg);
    }
    */

    // // 手动选择初始目标框
    // Rect bbox = selectROI("选择目标框", frame, false, false);
    // if (bbox.width == 0 || bbox.height == 0) {
    //     cout << "未选择有效的目标框" << endl;
    //     return -1;
    // }
    // destroyWindow("选择目标框");

    cv::Point predictedPoint;
    cv::Point measuredPoint;
    // 初始目标点
    cv::Point initialPoint(bbox.x + bbox.width/2, bbox.y + bbox.height/2); // 
    KalmanTracker kalTracker(initialPoint);
    // KalmanTracker kalTracker(cv::Point{960, 540});


    cout << "frame.size() : " << frame.size() << endl;

#if CV_KCF
    // 创建 KCF 跟踪器参数对象
    cv::TrackerKCF::Params params;

    // 配置特征类型  同时启用 desc_pca 和 desc_npca，实现多特征融合
    params.desc_npca = TrackerKCF::MODE::GRAY | TrackerKCF::MODE::CN;   // 使用 GRAY 特征
    params.desc_pca = TrackerKCF::MODE::GRAY | TrackerKCF::MODE::CN;   // 使用颜色特征
    //params.desc_pca = TrackerKCF::MODE::CUSTOM;
    
    params.desc_npca = TrackerKCF::MODE::GRAY;   // 使用 GRAY 特征
    params.desc_pca = TrackerKCF::MODE::CN;   // 使用颜色特征

    // params.detect_thresh = 0.5;
    params.compressed_size = 2;

    params.resize = true;
#define KCF_ROI_SIZE 80*80 //64 *64 // 80*80 // 256*256 90ms // 160*160 90ms // 80*80 20ms
#define KCF_ROI_SIZE_SMALL 40*40 //40*40
    params.max_patch_size = KCF_ROI_SIZE;
    std::cout << "Default Params:" << std::endl;
    std::cout << "detect_thresh: " << params.detect_thresh << std::endl;//0.5
    std::cout << "sigma: " << params.sigma << std::endl;//0.2
    std::cout << "lambda: " << params.lambda << std::endl;//0.0001
    std::cout << "interp_factor: " << params.interp_factor << std::endl;//0.075  //控制更新速度
    std::cout << "output_sigma_factor: " << params.output_sigma_factor << std::endl;//0.0625
    std::cout << "pca_learning_rate: " << params.pca_learning_rate << std::endl;//0.15
    std::cout << "resize: " << params.resize << std::endl;//1
    std::cout << "split_coeff: " << params.split_coeff << std::endl;//1
    std::cout << "wrap_kernel: " << params.wrap_kernel << std::endl;//0
    std::cout << "compress_feature: " << params.compress_feature << std::endl;//1
    std::cout << "max_patch_size: " << params.max_patch_size << std::endl;
    std::cout << "compressed_size: " << params.compressed_size << std::endl;//2
    std::cout << "desc_pca: " << params.desc_pca << std::endl;//2
    std::cout << "desc_npca: " << params.desc_npca << std::endl;//1

    // 初始化 KCF 跟踪器
    cv::Ptr<TrackerKCF> tracker = TrackerKCF::create();//params
    // cv::Ptr<TrackerCSRT> tracker = TrackerCSRT::create();
    std::cout << "bbox.area: " << bbox.area() << std::endl;
#if ROI_RESIZE

    Point const_bbox{64, 64};//将roi区域缩放，计算图像整体缩放比例，roi 缩放后位置
    float fx = const_bbox.x * 1.0 / bbox.width;
    float fy = const_bbox.y * 1.0 / bbox.height;

    cv::resize(frame, frame_sub, cv::Size(), fx, fy, cv::INTER_LINEAR);
    //cv::imshow("frame_sub", frame_sub);
    cv::Mat frame_sub_show = frame_sub.clone();
    bbox_sub = cv::Rect(bbox.x*fx, bbox.y*fy, bbox.width*fx, bbox.height*fy);
    cv::rectangle(frame_sub_show, bbox_sub, cv::Scalar(0, 150, 122));
    cv::imshow("frame_sub_show", frame_sub_show);
    waitKey(0);

    tracker->init(frame_sub, bbox_sub);
#else
    tracker->init(frame, bbox);
#endif
#else
    bool LAB = false;
    bool HOG = true;
    bool FIXEDWINDOW = true;
#define KCF
#ifndef KCF
    bool MULTISCALE = false; // true
    KCFTracker *tracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
#else
    bool MULTISCALE = true;
    FDSSTTracker *tracker = new FDSSTTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
#endif


    tracker->init(bbox, frame);

    // cv::Ptr<TrackerCSRT> tracker_csrt = TrackerCSRT::create();
    // tracker_csrt->init(frame, bbox);
    cout << frame.channels() << endl;
    // cv::waitKey(0);
#endif
    ///
// 15342513247

    //tracker.maxResponse 
    // 保存最后一帧的目标框位置
    cv::Rect lastbbox = bbox;


    Mat templateImage; // 模板图像
    //    queue<cv::Mat> templateImg_queue;
    Mat showFrame;

    // 初始化队列，最大容量为 30
    std::deque<FrameData> frameQueue;
    const int MAX_SIZE = 60;
    cv::Mat maxValueImage;


        // 模拟：准备一些简单训练数据（此处需改进为真实目标样本提取）
    vector<Mat> trainSamples;
    vector<int> trainLabels;
    cv::Ptr<cv::ml::SVM> svmModel;

#if !IMG_SEQ
    while (true) {
        // 读取下一帧
        if (!cap.read(frame)) {
            cout << "视频结束或无法读取帧" << endl;
            break;
        }
#if GRAY_INPUT
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
#endif


// #if CV_KCF
//         bool m_tracker_success = tracker->update(frame, bbox);// 更新跟踪器
// #else
//         bool m_tracker_success = tracker->update(frame, bbox, 0.5);
// #endif
//         if (m_tracker_success) {
//             // 绘制跟踪结果
//             rectangle(frame, bbox, Scalar(0, 255, 0), 2, 1);
//             putText(frame, "Tracking", Point(bbox.x, bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
//         } else {
//             // 跟踪失败
//             putText(frame, "Tracking failure detected", Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
//         }

//         showScaleImg("KCF Tracking", frame);
//         waitKey(0);

//         // 按 'q' 退出
//         if (waitKey(1) == 'q') {
//             break;
//         }
//     }
//     cap.release();
        static int i = 0;
        i++;

        showFrame = frame.clone(); // 显示
        if(showFrame.channels() == 1) {
            cv::cvtColor(showFrame, showFrame, cv::COLOR_GRAY2BGR);
        }
        if (frame.empty()) {
            cout << "无法加载图片" << endl;
            continue;
        }
#else

    // 逐帧读取图片并进行目标跟踪
    for (size_t i = m_frameStartId + 1; i < imageFiles.size(); ++i) {
#if GRAY_INPUT
        frame = imread(imageFiles[i], cv::IMREAD_GRAYSCALE);// gray
        // cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
#else
        frame = imread(imageFiles[i]);// color
#endif


        showFrame = frame.clone(); // 显示
        if(showFrame.channels() == 1) {
            cv::cvtColor(showFrame, showFrame, cv::COLOR_GRAY2BGR);
        }
        if (frame.empty()) {
            cout << "无法加载图片：" << imageFiles[i] << endl;
            continue;
        }

        // std::cout << "frame id: " << i << std::endl;


#endif
        time_ms.start();
        // 更新跟踪器
#if CV_KCF

#if ROI_RESIZE
    cv::resize(frame, frame_sub, cv::Size(), fx, fy, cv::INTER_LINEAR);
    bool m_tracker_success = tracker->update(frame_sub, bbox_sub);
    bbox = cv::Rect(bbox_sub.x/fx, bbox_sub.y/fy, bbox_sub.width/fx, bbox_sub.height/fy);
#else
    bool m_tracker_success = tracker->update(frame, bbox);
#endif

#else
    bool m_tracker_success;
    if(re_detect_cnt > 0) {
        m_tracker_success = false;
        nonblocking_delay_us(5 * 1000);
    } else
#ifndef KCF
        m_tracker_success = tracker->update(frame, bbox, 0.5);
        static int cnt = 0;
        cnt ++;
        if(cnt == 50) {
            // waitKey(0);
            // tracker->m_padding = 5;
        }
        
#else
        // bbox = tracker->update(frame);
        // m_tracker_success = true;
        m_tracker_success = tracker->update(frame, bbox);
#endif
        // tracker_csrt->update(frame, bbox_csrt);
#endif
        
        time_ms.stop();
        //float score = tracker->getTrackingScore();
        
        std::cout << "frame id: " << i <<
            "m_tracker_success: " << m_tracker_success <<
            "  time = " << time_ms.getTimeMilli() << "ms" <<
            std::endl;
        time_ms.reset();
        // waitKey(0);

        // 使用 Kalman 滤波器预测位置
        predictedPoint = kalTracker.predict();
        predictedPoint = clampPoint(predictedPoint, frame);
        // 绘制预测位置
        circle(showFrame, predictedPoint, 5, Scalar(255, 0, 0), -1); // 蓝色
        // cout << "predictedPoint : " << predictedPoint << endl;


        // detector->detect(frame, results);
        results[0].classId = 1;

        if (m_tracker_success) {
            //  
            templateImage = frame(bbox);//实时更新的跟踪模板，效果变差
            // templateImg_queue.push(templateImage);
            // while(templateImg_queue.size()>20) // 30
            // {templateImg_queue.pop();}
            if(tracker->m_apceValue > 40 || frameQueue.size() == 0) { // 重新初始化后第一帧必须放进去
                // 添加到队列
                frameQueue.push_back(FrameData(templateImage, tracker->m_apceValue));
            }
            // 如果队列超出大小，移除最旧的元素
            if (frameQueue.size() > MAX_SIZE) {
                frameQueue.pop_front();
            }

            // 复制队列内容到 vector 以便排序
            std::vector<FrameData> sortedQueue(frameQueue.begin(), frameQueue.end());
            std::sort(sortedQueue.begin(), sortedQueue.end(), compareFrameData);
            // cout << "sortedQueue " << sortedQueue.size() << endl;
            if(sortedQueue.size() == 0) std::cerr << "error : sortedQueue.size == 0..." << std::endl;

            // 提取 m_apceValue 最大的图像
            maxValueImage = sortedQueue.back().image;
            std:: cout << "sortedQueue.back().m_apceValue: " << sortedQueue.back().m_apceValue << std::endl;

            // cout << templateImage.type();
            // imshow("templateImage", templateImage);
            measuredPoint.x = bbox.x + bbox.width / 2;
            measuredPoint.y = bbox.y + bbox.height / 2;

            // 更新测量值
            kalTracker.update(measuredPoint);

            if(tracker->m_apceValue > 40 && 0) {
                trainSamples.push_back(frame(bbox));     trainLabels.push_back(10);
                trainSamples.push_back(frame(Rect(bbox.tl().x + bbox.width, bbox.tl().y, bbox.width, bbox.height))); trainLabels.push_back(-1);
                trainSamples.push_back(frame(Rect(bbox.tl().x - bbox.width, bbox.tl().y, bbox.width, bbox.height))); trainLabels.push_back(-1);
                trainSamples.push_back(frame(Rect(bbox.tl().x, bbox.tl().y + bbox.height, bbox.width, bbox.height))); trainLabels.push_back(-1);
                trainSamples.push_back(frame(Rect(bbox.tl().x, bbox.tl().y - bbox.height, bbox.width, bbox.height))); trainLabels.push_back(-1);

                svmModel = trainSVM(trainSamples, trainLabels);
                trainSamples.clear();
                trainLabels.clear();
            }

            // std::cout << "showFrame.channels: " << showFrame.channels() << std::endl; 
            // imshow("showFrame1", showFrame);
            // waitKey(0);
            // std::cout << "bbox.tl: " << bbox.tl() << std::endl;
            // std::cout << "bbox.size: " << bbox.size() << std::endl;

            if (showFrame.empty()) {
                std::cerr << "showFrame is empty after cvtColor!" << std::endl;
                return 0;
            }
            
            // 绘制跟踪结果
            // rectangle(showFrame, bbox, Scalar(0, 255, 0), 2, 1);
            drawCornerBox(showFrame, bbox, 1, Scalar(0, 255, 0));

            rectangle(showFrame, bbox_csrt, Scalar(255, 0, 255), 2, 1);

            cv::circle(showFrame, Point(measuredPoint.x, measuredPoint.y), 4, cv::Scalar(0, 0, 255), -1);
            if(m_retest_cnt) putText(showFrame, "Retest Tracking" + to_string(m_retest_cnt)+": "+to_string(bbox.width)+"*"+to_string(bbox.height), Point(bbox.x, bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            else putText(showFrame, "Tracking: "+to_string(bbox.width)+"*"+to_string(bbox.height), Point(bbox.x, bbox.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            // maxValueImage = templateImg_queue.front();
            imshow("queue", maxValueImage);

            // putText(showFrame, "m_apceValue: "+to_string(tracker->m_apceValue), Point(800, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
            // putText(showFrame, "m_psr       : "+to_string(tracker->m_psr), Point(800, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
            // putText(showFrame, "m_psr/apce : "+to_string(tracker->m_psr/tracker->m_apceValue), Point(800, 150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);

            // imshow("showFrame1", showFrame);
            
            re_detect_cnt = 0;

            results[0].confidence = 1;
            results[0].box = bbox;

        } else {
#if 1
            putText(showFrame, "Tracking failure detected", Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);

            // cv::Rect bbox_svm = reDetect(frame, svmModel, bbox.size());
            // rectangle(showFrame, bbox_svm, Scalar(100, 100, 255), 2, 1);


            // maxValueImage = templateImg_queue.front();

            // 重检测计数
            // static int re_detect_cnt = 0;
if(re_detect_cnt >= 0) {
            rec_time_ms.start();
            // 重新检测
            //1.搜索范围 整个图像
            //Mat currentFrame = frame;
            //时间越久，范围扩大
            //2.设置搜索范围
            Point2f curV = kalTracker.getVelocity();
            int mulx_, muly_;
            // cout << "curV.x = " << curV.x << " curV.y = " << curV.y << endl;
            if(abs(curV.x) > abs(curV.y)) {
                muly_ = 2;
                mulx_ = (int)abs(2 * curV.x / curV.y);
                mulx_ = std::max(std::min(mulx_, 6), 3);
            } else {
                mulx_ = 2;
                muly_ = (int)abs(2 * curV.y / curV.x);
                muly_ = std::max(std::min(muly_, 6), 3);
            }
            // cout << "mulx_ = " << mulx_ << " muly_ = " << muly_ << endl;
            Rect range = clampROI(predictedPoint, Size(maxValueImage.cols * mulx_, maxValueImage.rows * muly_), frame);
            // Rect range = clampROI(predictedPoint, Size(maxValueImage.cols * 2, maxValueImage.rows * 2), frame);

            //Rect range = clampROI(predictedPoint, Size(initRecImg.cols * 2, initRecImg.rows * 2), frame);
            
            // rectangle(showFrame, range, Scalar(255, 0, 100), 2, 1);
            drawCornerBox(showFrame, range, 1, Scalar(255, 0, 100));

            Mat currentFrame = frame(range).clone(); // keep continuous

            double maxVal;
            cv::Rect detectedROI;
#if 1
            if(frame.channels() == 3) {
                cv::Mat currentFrameGray, templateImg_queueGray;
                if(currentFrame.empty()) std::cerr << "error : currentFrame is empty..." << std::endl;
                cv::cvtColor(currentFrame, currentFrameGray, cv::COLOR_BGR2GRAY); // 不连续 连续
                cv::cvtColor(maxValueImage, templateImg_queueGray, cv::COLOR_BGR2GRAY); // 连续 连续
                cout << "frame.channels() == " << frame.channels() << endl;

                // waitKey(0);
                detectedROI = detectMatch_(currentFrameGray, templateImg_queueGray, maxVal); // _
                // detectedROI = detectAndMatch(templateImg_queueGray, currentFrameGray, maxVal);
                // detectedROI = detectMatch(currentFrameGray(detectedROI), templateImg_queueGray, maxVal);
            } else {
                // waitKey(0);
                // if(!currentFrame.isContinuous()) {
                //     currentFrame = currentFrame.clone();
                // }

                detectedROI = detectMatch_(currentFrame, maxValueImage, maxVal); // maxValueImage is continuous
                // detectedROI = detectAndMatch(maxValueImage, currentFrame, maxVal);
            }
#else
            // cv::Rect detectedROI = detectMatch(currentFrame, maxValueImage, maxVal);
            // // Rect detectedROI = detectMatch(currentFrame, templateImg_queue.front(), maxVal);
            detectedROI = reDetect(currentFrame, svmModel, bbox.size(), maxVal);
            maxVal *= 0.4;
#endif
            detectedROI.x = range.x + detectedROI.x;
            detectedROI.y = range.y + detectedROI.y;

            // detectedROI = clampRect(detectedROI, frame);
            detectedROI &= Rect(0, 0, frame.cols, frame.rows);
            // boundingBoxResult = Rect(x1, y1, x2 - x1, y2 - y1) & Rect(Point(0, 0), image.size());

            rectangle(showFrame, detectedROI, Scalar(0, 0, 255), 2, 1);
            // drawCornerBox(showFrame, detectedROI, 1,  Scalar(0, 0, 255));

            //imshow("queue"+to_string(m_retest_cnt), maxValueImage);


            static cv::Point lastDetectedPoint;
            rec_time_ms.stop();
            std::cout <<"detected time = " << rec_time_ms.getTimeMilli() << "ms" << std::endl;
            rec_time_ms.reset();
            // waitKey(0);

            results[0].confidence = maxVal;
            results[0].box = detectedROI;


            // if(maxVal > 0.999) {
            //     m_retest = true;
            //     m_match_cnt = 10; // 特征点匹配
            // }

            re_detect_cnt ++;
            if(re_detect_cnt > 90) {
                cout << "停止重检测" << endl;
                re_detect_cnt = -1;
                cv::waitKey(0);
            }
            if(maxVal > 0.7) {
                m_retest = true;
                m_match_cnt = m_match_cnt + 4; // 6
            }

            if(maxVal > 0.6)//0.52 //0.55
            {
                m_match_cnt ++;
                if(maxVal > 0.75) m_match_cnt ++;
                int err_x = lastDetectedPoint.x - detectedROI.x;// cv::Point err_point = lastDetectedPoint - detectedROI.tl();
                int err_y = lastDetectedPoint.y - detectedROI.y;

                if(abs(err_x)>detectedROI.width/4 || abs(err_y)>detectedROI.height/4) {
                    m_match_cnt = 0;
                }
                m_retest = true;
                cout << "m_match_cnt:" << m_match_cnt << endl;
                //cout << "err_x: " << err_x << " err_y: " << err_y << endl;
                lastDetectedPoint = detectedROI.tl();
            } else {
                m_retest = false;
            }
            // 重检测成功，重新初始化 KCF 跟踪器
            if(m_retest && (m_match_cnt>7) /*&& detectedROI.width>0 && detectedROI.height>0*/ ) {// 5
                m_match_cnt = 0;
                m_retest_cnt ++;
                re_detect_cnt = 0;

                cout << "retest_cnt:......................." << m_retest_cnt << endl;
                // 替换为空队列
                // std::queue<cv::Mat> empty;
                // templateImg_queue.swap(empty);

                std::deque<FrameData> emptyDeque;
                frameQueue.swap(emptyDeque);

                #if CV_KCF
                TrackerKCF::Params params;
                params.max_patch_size = KCF_ROI_SIZE;

                if(detectedROI.area() < KCF_ROI_SIZE)
                    params.max_patch_size = KCF_ROI_SIZE_SMALL;

                // std::cout << "params.resize: " << params.resize << std::endl;
                // waitKey(0);
                tracker = TrackerKCF::create(params);
                // tracker = TrackerCSRT::create();
                // getOptimalROISize(detectedROI);
                tracker->init(frame, detectedROI);
                #else
                if(tracker) delete tracker;
                #ifndef KCF
                tracker = new KCFTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
                #else
                tracker = new FDSSTTracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
                #endif
                bbox = detectedROI; // 重新初始化跟踪器

                tracker->init(bbox, frame);
                #endif
                cout << "m_retest_cnt: "<< m_retest_cnt << "detectedROI: " << detectedROI << endl;

                /*
                // 若不重新初始化跟踪器，仅更新 ROI 位置
                bbox.x = detectedROI.x; bbox.y = detectedROI.y;
                */

                //templateImg_queue.push(frame(bbox));
                ////imshow("decROI"+to_string(m_retest_cnt), frame(bbox));
                //if(m_retest_cnt==1) waitKey(0);
                rectangle(showFrame, bbox, Scalar(100, 100, 255), 2, 1);
                // cout << "init" << endl;
                // waitKey(0);

            }
#endif
        }
        }

        test_bytetrack(showFrame, results, bytetracker);

        // rectangle(showFrame, Rect{predictedPoint.x - bbox.width/2, predictedPoint.y - bbox.height/2, bbox.width, bbox.height},
        // Scalar(0, 255, 255), 2, 1); // 滤波预测框 黄色

        putText(showFrame, "m_apceValue: "+to_string(tracker->m_apceValue), Point(800, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
        putText(showFrame, "m_psr       : "+to_string(tracker->m_psr), Point(800, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);
        putText(showFrame, "m_psr/apce : "+to_string(tracker->m_psr/tracker->m_apceValue), Point(800, 150), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 255, 255), 2);

        showScaleImg("KCF Tracking", showFrame);
        if (waitKey(30) == 'q') { // 按 'q' 键退出
            break;
        }
    }
// #endif

    // destroyAllWindows();
    return 0;
}

