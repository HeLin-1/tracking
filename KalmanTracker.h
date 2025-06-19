#ifndef _KALMAN_TRACKER_H_
#define _KALMAN_TRACKER_H_

#include <opencv2/opencv.hpp>

class KalmanTracker {
public:
    KalmanTracker(cv::Point initialPoint) {
        // 初始化卡尔曼滤波器
        int stateSize = 4;  // 状态变量: x, y, vx, vy
        int measSize = 2;   // 观测变量: x, y
        int contrSize = 0;  // 控制变量: 无
        kf = cv::KalmanFilter(stateSize, measSize, contrSize, CV_32F);

        // 状态转移矩阵 (A)
        kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 
            1, 0, 1, 0, 
            0, 1, 0, 1, 
            0, 0, 1, 0, 
            0, 0, 0, 1);

        // 观测矩阵 (H)
        kf.measurementMatrix = (cv::Mat_<float>(2, 4) << 
            1, 0, 0, 0, 
            0, 1, 0, 0);

        // 噪声协方差矩阵
        cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-4));      // 过程噪声 (Q)
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));  // 观测噪声 (R)
        cv::setIdentity(kf.errorCovPost, cv::Scalar(1));            // 初始误差协方差 (P)

        // 初始化状态
        kf.statePost = (cv::Mat_<float>(4, 1) << initialPoint.x, initialPoint.y, 0, 0); // 初始位置 初始速度 0
        // cout << "kf.statePost init: " << kf.statePost << endl;
    }

    // 预测下一时刻的点
    cv::Point predict() {
        cv::Mat prediction = kf.predict();
        m_predictV = {prediction.at<float>(2), prediction.at<float>(3)};
        return cv::Point(static_cast<int>(prediction.at<float>(0)), 
                        static_cast<int>(prediction.at<float>(1)));
    }

    // 更新测量值并校正
    void update(cv::Point measuredPoint) {
        cv::Mat measurement = (cv::Mat_<float>(2, 1) << measuredPoint.x, measuredPoint.y);
        kf.correct(measurement); // 卡尔曼滤波器更新
        // cout << "kf.statePost : " << kf.statePost << endl;
    }

    // 单独获取当前估计的速度（如果需要在更新后获取速度）
    cv::Point2f getVelocity() const {
        return m_predictV;
    }

private:
    cv::KalmanFilter kf;
    cv::Point2f m_predictV;
};

#endif // KALMAN_TRACKER_H

/*
3. 参数确定方法
(1) 关键参数说明
参数	含义	调参建议
transitionMatrix	状态转移矩阵（运动模型）	匀速模型固定为 [1,0,1,0; 0,1,0,1; 0,0,1,0; 0,0,0,1]
processNoiseCov	过程噪声（模型不确定性）	越小越信任模型（如 1e-4~1e-2）
measurementNoiseCov	观测噪声（检测误差）	越小越信任检测（如 1e-1~1e0）
errorCovPost	初始状态协方差（初始不确定性）	通常设为单位矩阵 np.eye(4)
(2) 调参技巧

    过程噪声（processNoiseCov）：

        目标运动越规律（如匀速），噪声越小（如 1e-4）。

        目标运动随机性大（如人行走），噪声调大（如 1e-2）。

    观测噪声（measurementNoiseCov）：

        检测器精度高（如 YOLOv8），噪声设小（如 1e-1）。

        检测器抖动大，噪声设大（如 1e0）。

*/