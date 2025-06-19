/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */

#ifndef FDSSTTRACKER_CPP
#define FDSSTTRACKER_CPP
#include <time.h>

#include "fdssttracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"

#include "fhog.h"

#include "labdata.hpp"

// #define PFS_DEBUG

template <typename T>
cv::Mat rangeToColVector(int begin, int end, int n)
{
    cv::Mat_<T> colVec(1, n);

    for (int i = begin, j = 0; i <= end; ++i, j++)
        colVec.template at<T>(0, j) = static_cast<T>(i);

    return colVec;
}

template <typename BT, typename ET>
cv::Mat pow(BT base_, const cv::Mat_<ET> &exponent)
{
    cv::Mat dst = cv::Mat(exponent.rows, exponent.cols, exponent.type());
    int widthChannels = exponent.cols * exponent.channels();
    int height = exponent.rows;

    // http://docs.opencv.org/doc/tutorials/core/how_to_scan_images/how_to_scan_images.html#the-efficient-way
    if (exponent.isContinuous())
    {
        widthChannels *= height;
        height = 1;
    }

    int row = 0, col = 0;
    const ET* exponentd = 0;
    ET* dstd = 0;

    for (row = 0; row < height; ++row)
    {
        exponentd = exponent.template ptr<ET>(row);
        dstd = dst.template ptr<ET>(row);

        for (col = 0; col < widthChannels; ++col)
        {
            dstd[col] = std::pow(base_, exponentd[col]);
        }
    }

    return dst;
}

void shift(const cv::Mat& src, cv::Mat& dst, cv::Point2f delta, int fill, cv::Scalar value = cv::Scalar(0, 0, 0, 0)) {
    // error checking
    CV_Assert(fabs(delta.x) < src.cols && fabs(delta.y) < src.rows);

    // split the shift into integer and subpixel components
    cv::Point2i deltai(static_cast<int>(ceil(delta.x)), static_cast<int>(ceil(delta.y)));
    cv::Point2f deltasub(fabs(delta.x - deltai.x), fabs(delta.y - deltai.y));

    // INTEGER SHIFT
    // first create a border around the parts of the Mat that will be exposed
    int t = 0, b = 0, l = 0, r = 0;
    if (deltai.x > 0) l = deltai.x;
    if (deltai.x < 0) r = -deltai.x;
    if (deltai.y > 0) t = deltai.y;
    if (deltai.y < 0) b = -deltai.y;
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, t, b, l, r, fill, value);

    // SUBPIXEL SHIFT
    float eps = std::numeric_limits<float>::epsilon();
    if (deltasub.x > eps || deltasub.y > eps) {
        switch (src.depth()) {
        case CV_32F:
        {
            cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
            cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
            sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
            break;
        }
        case CV_64F:
        {
            cv::Matx<double, 1, 2> dx(1 - deltasub.x, deltasub.x);
            cv::Matx<double, 2, 1> dy(1 - deltasub.y, deltasub.y);
            sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
            break;
        }
        default:
        {
            cv::Matx<float, 1, 2> dx(1 - deltasub.x, deltasub.x);
            cv::Matx<float, 2, 1> dy(1 - deltasub.y, deltasub.y);
            padded.convertTo(padded, CV_32F);
            sepFilter2D(padded, padded, CV_32F, dx, dy, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
            break;
        }
        }
    }

    // construct the region of interest around the new matrix
    cv::Rect roi = cv::Rect(std::max(-deltai.x, 0), std::max(-deltai.y, 0), 0, 0) + src.size();
    dst = padded(roi);
}

// Constructor
FDSSTTracker::FDSSTTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5; // 2.5
    // output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;

    if (hog)
    { // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6;
        // TPAMI
        // interp_factor = 0.02;
        // sigma = 0.5;
        cell_size = 4;
        _hogfeatures = true;

        num_compressed_dim = 13;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4;
            // output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &Data);
            cell_sizeQ = cell_size * cell_size;
        }
        else {
            _labfeatures = false;
        }
    }
    else
    { // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }

    if (multiscale) { // multiscale
        template_size = 96;
        // scale parameters initial
        scale_padding = 1.0;
        scale_step = 1.02; // 1.02
        scale_sigma_factor = 1.0 / 16;

        n_scales = 11; // 3 4 5 不行 9
// 6(189)  8(183)9(175) 10(170) 11(157) 13(150) 33(105)- 60(79.19)   9 12 与时长正相关
        n_interp_scales = 55; // 11 // 33 // 50 55

        scale_lr = 0.025;
        scale_max_area = 512;
        currentScaleFactor = 1;
        scale_lambda = 0.01;

        if (!fixed_window) {
            // printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) { // fit correction without multiscale
        template_size = 96;
        // template_size = 100;
        scale_step = 1;
    }
    else {
        template_size = 1;
        scale_step = 1;
    }
}

// Initialize tracker
void FDSSTTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);

    // std::cout << "size_patch[0]: " << size_patch[0] << std::endl; // 26 
    // std::cout << "size_patch[1]: " << size_patch[1] << std::endl; // 26
    // // 在程序初始化时调用（确保线程池初始化）
    // FFTTools::init_fft_pool(size_patch[0], size_patch[1]); // fft 尺寸 e.g 26x26

    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));

    dsstInit(roi, image);
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
}

// Update position based on the new frame
cv::Rect FDSSTTracker::update(cv::Mat image)
{

}
bool FDSSTTracker::update(cv::Mat image, cv::Rect &bbox)
{
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value;

#ifdef PFS_DEBUG
    cv::TickMeter t1_ms, t2_ms, t3_ms;
    t1_ms.start();
#endif
    cv::Point2f res = detect(getFeatures(image, 0, 1.0f), peak_value);
#ifdef PFS_DEBUG
    t1_ms.stop();
    std::cout << "translation detction duration: " << t1_ms.getTimeMilli() << "ms \n";
    t1_ms.reset();
#endif
    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale * currentScaleFactor);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale * currentScaleFactor);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;


    // 峰值小于设定阈值，认为跟踪失败
    // printf("peak_value: %f\n", peak_value);
    if (m_apceValue < 12 && peak_value < 0.8 || m_psr < 8.0) // 10 0.8
    {
        // bbox = _roi;
        std::cout << "tracking failed....." << std::endl;
        return false;
    }

        // Update scale

#ifdef PFS_DEBUG
    t3_ms.start();
#endif
    cv::Point2i scale_pi = detect_scale(image);
#ifdef PFS_DEBUG
    t3_ms.stop();
    std::cout << "scale detction duration: " << t3_ms.getTimeMilli() << "ms \n";
    t3_ms.reset();
#endif
    currentScaleFactor = currentScaleFactor * interp_scaleFactors[scale_pi.x];
    if (currentScaleFactor < min_scale_factor)
        currentScaleFactor = min_scale_factor;
    // else if(currentScaleFactor > max_scale_factor)
    //   currentScaleFactor = max_scale_factor;

    train_scale(image);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);
    cv::Mat x = getFeatures(image, 0);

    // train(x, interp_factor);

    float _factor = std::min(1.f, exp(4.f * (peak_value - 0.6f)));
    std::cout << "peak_value = " << peak_value << std::endl;
    // static double pre_psr = m_psr;
    if(m_psr < 15 || peak_value < 0.2) { // 15
        _factor = 0;
        std::cout << ".........................不更新......\n";
    }
    // pre_psr = m_psr;
    if(_factor > 1e-6) {
        train(x, interp_factor * _factor);
    }

    bbox = cv::Rect(_roi) & cv::Rect(0, 0, image.cols, image.rows);
    return true;
}

// Detect the new scaling rate
cv::Point2i FDSSTTracker::detect_scale(cv::Mat image)
{
    cv::Mat xsf = FDSSTTracker::get_scale_sample(image);

    // Compute AZ in the paper
    cv::Mat add_temp;
    cv::reduce(FFTTools::complexMultiplication(sf_num, xsf), add_temp, 0, cv::REDUCE_SUM);

    // compute the final y
    cv::Mat scale_responsef = FFTTools::complexDivisionReal(add_temp, (sf_den + scale_lambda));

    cv::Mat interp_scale_responsef = resizeDFT(scale_responsef, n_interp_scales);

    cv::Mat interp_scale_response;
    // std::cout << "interp_scale_responsef size : " << interp_scale_responsef.size() << std::endl; // [33 x 1]
    // std::cout << "interp_scale_responsef type : " << interp_scale_responsef.type() << std::endl; // 13

    cv::idft(interp_scale_responsef, interp_scale_response);

    interp_scale_response = FFTTools::real(interp_scale_response);

    // Get the max point as the final scaling rate
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(interp_scale_response, NULL, &pv, NULL, &pi);

    return pi;
}

float computeAPCE(const cv::Mat& response, double &maxVal, cv::Point &maxLoc) {
    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);
    // response.type() CV_32F
    float minVal_F = static_cast<float>(minVal);
    cv::Mat temp;
    cv::pow(response - minVal_F, 2, temp);
    double meanVal = cv::mean(temp)[0];

    double apce = std::pow(maxVal - minVal, 2) / (meanVal + 1e-6); // 防止除0
    return static_cast<float>(apce);
}

// lin_edit
// im.size [24 x 24]
void FDSSTTracker::getAPCE(const cv::Mat &im, float &apce, double &maxVal, cv::Point &maxLoc)
{
    apce = computeAPCE(im, maxVal, maxLoc);
    // std::cout << "Apce = " << std::setprecision(10) << std::setw(15) << apce << std::endl;
    // std::cout << "minLoc = " << minLoc << "maxLoc = " << maxLoc << std::endl;
    std::cout << "APCE = " << apce << ", maxVal = " << maxVal << std::endl;

    int size = 5;  // 中心窗口大小
    cv::Rect excludeRegion(maxLoc.x - size / 2, maxLoc.y - size / 2, size, size);
    excludeRegion &= cv::Rect(0, 0, im.cols, im.rows);
    // 创建 mask 排除主峰区域
    cv::Mat mask = cv::Mat::ones(im.size(), CV_8U);
    mask(excludeRegion).setTo(0);  // 把主峰周围设为 0

    // 计算侧瓣区域的均值和标准差
    cv::Scalar mean, stddev;
    cv::meanStdDev(im, mean, stddev, mask);

    // 计算 PSR
    m_psr = (maxVal - mean[0]) / stddev[0];
    std::cout << "PSR: " << m_psr << std::endl;
}

// Detect object in the current frame.
cv::Point2f FDSSTTracker::detect(cv::Mat x, float &peak_value)
{
    using namespace FFTTools;

// #define PFS_DEBUG1
#ifdef PFS_DEBUG1
    cv::TickMeter t2_ms;
    t2_ms.start();
#endif
    x = features_projection(x);
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "x = features_projection(x): " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif
#ifdef PFS_DEBUG1
    t2_ms.start();
#endif
    cv::Mat z = features_projection(_tmpl);
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "z = features_projection(_tmpl): " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif
#ifdef PFS_DEBUG1
    t2_ms.start();
#endif
    cv::Mat k = gaussianCorrelation(x, z);
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "**************gaussianCorrelation duration: " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif

#ifdef PFS_DEBUG1
    t2_ms.start();
#endif

    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "complexMultiplication *******************: " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif
    // minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;

    // cv::Mat show;
    // cv::normalize(res, show, 0, 255, cv::NORM_MINMAX);
    // show.convertTo(show, CV_8UC1);
    // cv::Mat colorMapImg;
    // cv::applyColorMap(show, colorMapImg, cv::COLORMAP_JET);
    // FFTTools::showScaleImg("res", colorMapImg, 0.1);
    // std::cout << "res size = " << res.size() << std::endl;

#ifdef PFS_DEBUG1
    t2_ms.start();
#endif
    getAPCE(res, m_apceValue, pv, pi);
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "getAPCE : " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif
    peak_value = (float)pv;

#ifdef PFS_DEBUG1
    t2_ms.start();
#endif
    // subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols - 1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (pi.y > 0 && pi.y < res.rows - 1) {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;
#ifdef PFS_DEBUG1
    t2_ms.stop();
    std::cout << "subPixelPeak : " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif
    return p;
}

// train tracker with a single image
void FDSSTTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x;

    cv::Mat W, U, VT, X, out;

    X = _tmpl * _tmpl.t();
    cv::SVD::compute(X, W, U, VT);

    VT.rowRange(0, num_compressed_dim).copyTo(proj_matrix);

    x = features_projection(x);

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));

    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf;
}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
#if 0
cv::Mat FDSSTTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;

#ifdef PFS_DEBUG
    double t_start1 = clock();
#endif

    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    // std::cout << "size_patch[2] " << size_patch[2] << std::endl; // size_patch[2] = 13
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i); // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);

            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);

            rearrange(caux);
            caux.convertTo(caux, CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }

#ifdef PFS_DEBUG
    t_end = clock();
    std::cout << "gaussianCorrelation computation A duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif

    cv::Mat d;
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);
#ifdef PFS_DEBUG
    t_end = clock();
    std::cout << "gaussianCorrelation computation B duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif
    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);

#ifdef PFS_DEBUG
    t_end = clock();
    std::cout << "gaussianCorrelation computation ALL duration: " << (t_end - t_start1) / CLOCKS_PER_SEC << "\n";
#endif

    return k;
}
#else

#include <omp.h>
cv::Mat FDSSTTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F);
    if (_hogfeatures) {
        int num_threads = 3; // omp_get_max_threads(); 
#define HUIZONG 1
#if HUIZONG
        std::vector<cv::Mat> c_locals(num_threads, cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F));
#endif
        // 设置 OpenMP 线程数
        omp_set_num_threads(num_threads);
        // std::cout << "max_threads = " << omp_get_max_threads() << std::endl;
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            // std::cout << "thread_id = " << thread_id << std::endl;
            cv::Mat caux(size_patch[0], size_patch[1], CV_32FC2); // 复数矩阵
            cv::Mat x1aux, x2aux, local_sum = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F); // 线程局部变量 预分配
    
            #pragma omp for nowait
            for (int i = 0; i < size_patch[2]; i++) {
                x1aux = x1.row(i).reshape(1, size_patch[0]).clone();  // 需要 clone 以确保数据独立
                x2aux = x2.row(i).reshape(1, size_patch[0]).clone();
    
                cv::Mat fft_x1 = fftd(x1aux);
                cv::Mat fft_x2 = fftd(x2aux);
                cv::mulSpectrums(fft_x1, fft_x2, caux, 0, true);
                caux = fftd(caux, true);
                rearrange(caux);
    
                local_sum += real(caux);
            }
#if HUIZONG
            // 线程结束后，汇总到 c_locals
            c_locals[thread_id] = local_sum;
#else
            // 使用原子操作或临界区汇总结果
            #pragma omp critical
            c += local_sum;
#endif
        }
#if HUIZONG
        // 汇总所有线程的计算结果
        for (const auto& c_local : c_locals) {
            cv::add(c, c_local, c); // 避免直接 +=
        }
#endif
    } else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }

    double x1_energy = cv::norm(x1, cv::NORM_L2SQR);  // 避免中间矩阵
    double x2_energy = cv::norm(x2, cv::NORM_L2SQR);

    cv::Mat d;
    cv::max((x1_energy + x2_energy - 2.0 * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

    cv::threshold(d, d, 10, 10, cv::THRESH_TRUNC); // 限制最大值
    cv::Mat k; // 计算高斯核 k
    cv::exp(-d / (sigma * sigma), k);
    return k;
}
#endif

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat FDSSTTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat FDSSTTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    if (inithann)
    {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;

        if (template_size > 1)
        {                             // Fit largest dimension to the given template size
            if (padded_w >= padded_h) // fit to width
                _scale = padded_w / (float)template_size;
            else
                _scale = padded_h / (float)template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else { // No template size given, use ROI size
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }

        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else { // Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width * currentScaleFactor;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height * currentScaleFactor;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

    cv::Mat FeaturesMap;
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    }
#ifdef PFS_DEBUG1
    cv::TickMeter t2_ms;
    t2_ms.start();
#endif
    // HOG features
    FeaturesMap = fhog(z, cell_size);

    FeaturesMap = FeaturesMap.reshape(1, z.cols * z.rows / (cell_size * cell_size));

    FeaturesMap = FeaturesMap.t();

#ifdef PFS_DEBUG1
    t2_ms.stop();
    if(!inithann && fabs(scale_adjust - 1.0) < 1e-6)
        std::cout << "HOG features: " << t2_ms.getTimeMilli() << "ms \n";
    t2_ms.reset();
#endif

    if (inithann) {
        size_patch[0] = z.rows / cell_size;
        size_patch[1] = z.cols / cell_size;
        size_patch[2] = num_compressed_dim;
        createHanningMats();
    }

    return FeaturesMap;
}

cv::Mat FDSSTTracker::features_projection(const cv::Mat &FeaturesMap)
{

    cv::Mat out;
    out = proj_matrix * FeaturesMap;

    out = hann.mul(out);

    return out;
}

// Initialize Hanning window. Function called only in the first frame.
void FDSSTTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j < size_patch[0] * size_patch[1]; j++) {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float FDSSTTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    // if (divisor == 0)
    //     return 0;
    if (std::fabs(divisor) < 1e-6) return 0.0f; // lin_edit

    return 0.5 * (right - left) / divisor;
}

// Initialization for scales
void FDSSTTracker::dsstInit(const cv::Rect &roi, cv::Mat image)
{
    // The initial size for adjusting
    base_width = roi.width;
    base_height = roi.height;

    // Guassian peak for scales (after fft)

    // ������ֵǰ�ĳ߶����У�����Ҫ��ȡ��߶�������һ��ֵ
    cv::Mat colScales =
        rangeToColVector<float>(-floor((n_scales - 1) / 2),
                                ceil((n_scales - 1) / 2), n_scales);

    colScales *= (float)n_interp_scales / (float)n_scales;

    cv::Mat ss;
    shift(colScales, ss,
          cv::Point(-floor(((float)n_scales - 1) / 2), 0),
          cv::BORDER_WRAP, cv::Scalar(0, 0, 0, 0));

    cv::Mat ys;

    float scale_sigma = scale_sigma_factor * n_interp_scales;

    exp(-0.5 * ss.mul(ss) / (scale_sigma * scale_sigma), ys);

    ysf = FFTTools::fftd(ys);

    s_hann = createHanningMatsForScale();

    // Get all scale changing rate
    scaleFactors = pow<float, float>(scale_step, colScales);

    // ������ֵ��ĳ߶�����
    cv::Mat interp_colScales =
        rangeToColVector<float>(-floor((n_interp_scales - 1) / 2),
                                ceil((n_interp_scales - 1) / 2), n_interp_scales);

    cv::Mat ss_interp;
    shift(interp_colScales, ss_interp,
          cv::Point(-floor(((float)n_interp_scales - 1) / 2), 0),
          cv::BORDER_WRAP, cv::Scalar(0, 0, 0, 0));

    interp_scaleFactors = pow<float, float>(scale_step, ss_interp);

    // Get the scaling rate for compressing to the model size
    float scale_model_factor = 1;
    if(base_width * base_height > scale_max_area)
    {
        scale_model_factor = std::sqrt(scale_max_area / (float)(base_width * base_height));
    }
    scale_model_width = (int)(base_width * scale_model_factor);
    scale_model_height = (int)(base_height * scale_model_factor);

    // Compute min and max scaling rate
    min_scale_factor = std::pow(scale_step,
                                std::ceil(std::log((std::fmax(5 / (float)base_width, 5 / (float)base_height) * (1 + scale_padding))) / 0.0086));
    max_scale_factor = std::pow(scale_step,
                                std::floor(std::log(std::fmin(image.rows / (float)base_height, image.cols / (float)base_width)) / 0.0086));

    train_scale(image, true);
}

// Train method for scaling
void FDSSTTracker::train_scale(cv::Mat image, bool ini)
{
    cv::Mat xsf = get_scale_sample(image);

    // Adjust ysf to the same size as xsf in the first time
    if(ini)
    {
        int totalSize = xsf.rows;
        ysf = cv::repeat(ysf, totalSize, 1);
    }

    // Get new GF in the paper (delta A)
    cv::Mat new_sf_num;
    cv::mulSpectrums(ysf, xsf, new_sf_num, 0, true);

    // Get Sigma{FF} in the paper (delta B)
    cv::Mat new_sf_den;
    cv::mulSpectrums(xsf, xsf, new_sf_den, 0, true);
    cv::reduce(FFTTools::real(new_sf_den), new_sf_den, 0, cv::REDUCE_SUM);

    if(ini)
    {
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    }
    else
    {
        // Get new A and new B
        cv::addWeighted(sf_den, (1 - scale_lr), new_sf_den, scale_lr, 0, sf_den);
        cv::addWeighted(sf_num, (1 - scale_lr), new_sf_num, scale_lr, 0, sf_num);
    }

    update_roi();
}

// Update the ROI size after training
void FDSSTTracker::update_roi()
{
    // Compute new center
    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    printf("%f\n", currentScaleFactor);

    // Recompute the ROI left-upper point and size
    _roi.width = base_width * currentScaleFactor;
    _roi.height = base_height * currentScaleFactor;

    _roi.x = cx - _roi.width / 2.0f;
    _roi.y = cy - _roi.height / 2.0f;
}

// Compute the F^l in the paper
#if 1
cv::Mat FDSSTTracker::get_scale_sample(const cv::Mat &image)
{

    cv::Mat xsf;   // output
    int totalSize; // # of features
    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    for (int i = 0; i < n_scales; i++) // 优化循环耗时
    {
        // Size of subwindow waiting to be detect
        float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
        float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

        // Get the subwindow
        cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
        cv::Mat im_patch_resized;

        // Scaling the subwindow
        if (scale_model_width > im_patch.cols)
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);
        else
            resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);

        // Compute the FHOG features for the subwindow
    //         if(im_patch_resized.channels() == 3) {
    //   cv::cvtColor(im_patch_resized, im_patch_resized, cv::COLOR_BGR2GRAY);
    // }
        cv::Mat hogs = fhog(im_patch_resized, cell_size);

        if (i == 0)
        {
            totalSize = hogs.cols * hogs.rows * 32;
            xsf = cv::Mat(cv::Size(n_scales, totalSize), CV_32F, float(0));
        }

        // Multiply the FHOG results by hanning window and copy to the output
        cv::Mat FeaturesMap = hogs.reshape(1, totalSize);
        float mul = s_hann.at<float>(0, i);
        FeaturesMap = mul * FeaturesMap;
        FeaturesMap.copyTo(xsf.col(i));
    }

    // Do fft to the FHOG features row by row
    xsf = FFTTools::fftd(xsf, 0, 1);

    return xsf;
}
#else
cv::Mat FDSSTTracker::get_scale_sample(const cv::Mat &image) {
        
    using namespace FFTTools;
    cv::Mat xsf; // 输出矩阵
    int totalSize = 0; // 初始化 totalSize 为 0

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    // Size of subwindow waiting to be detect
    float patch_width = base_width * scaleFactors[0] * currentScaleFactor;
    float patch_height = base_height * scaleFactors[0] * currentScaleFactor;
    // Get the subwindow
    cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
    cv::Mat im_patch_resized;
    // Scaling the subwindow
    if (scale_model_width > im_patch.cols)
        resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);
    else
        resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);


    // 在并行循环之前计算 totalSize
    // 假设 hogs 维度在调整大小后对于所有尺度都是恒定的
    // cv::Mat dummy_patch(scale_model_height, scale_model_width, CV_8UC1, cv::Scalar(0));
    cv::Mat dummy_hogs = fhog(im_patch_resized, cell_size);
    totalSize = dummy_hogs.cols * dummy_hogs.rows * dummy_hogs.channels(); // 如果 FHOG 有 32 个通道，则为 32
    // 在并行循环之前初始化 xsf
    xsf = cv::Mat(cv::Size(n_scales, totalSize), CV_32F, float(0));

    omp_set_num_threads(3);
    // #pragma omp parallel for// 私有变量隐式处理或显式定义
    #pragma omp parallel
    {
    #pragma omp for nowait
    for (int i = 0; i < n_scales; i++) {
        // 每个线程都有这些变量的私有副本（循环迭代器默认是私有的）
        float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
        float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

        // 获取子窗口
        cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
        cv::Mat im_patch_resized;

        // 如果 extractImage 返回空补丁，则进行处理
        if (im_patch.empty()) {
            // 决定如何处理这种情况：
            // - 用零填充 xsf.col(i)
            // - 记录警告
            // 目前，我们只是跳过此尺度的特征计算
            // 并将 xsf.col(i) 保持为零（由于初始分配）。
            continue;
        }

        // 缩放子窗口
        // 注意：'if' 条件 (scale_model_width > im_patch.cols) 在某些情况下总是真，
        // 在其他情况下总是假，这取决于模型尺寸与实际补丁尺寸的关系。
        // 它通常只是表示“调整大小到 scale_model_width/height”。
        resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);

        // 计算子窗口的 FHOG 特征
        cv::Mat hogs = fhog(im_patch_resized, cell_size);

        // 重塑并应用汉宁窗
        cv::Mat FeaturesMap = hogs.reshape(1, totalSize); // 重塑为 1 行，totalSize 列（或反之，取决于列主序）
        float mul = s_hann.at<float>(0, i);
        FeaturesMap = mul * FeaturesMap;

        // 复制到输出矩阵的列。如果 xsf 是共享的，这里是关键部分。
        // 然而，如果每个线程写入一个 *不同* 的列，那就是线程安全的。
        // 由于 'i' 是循环变量并直接映射到列索引，
        // 每个线程写入自己独特的列，因此此 `copyTo` 操作不需要显式同步（如 critical section）。
        // #pragma omp critical
        FeaturesMap.copyTo(xsf.col(i));
    }
    }

    // 对 FHOG 特征进行逐行 FFT
    // 如果 FFTTools::fftd 内部没有并行化，并且 xsf 很大，
    // 这可能是另一个 OpenMP 并行化的点，可能在行上使用 #pragma omp parallel for。
    xsf = FFTTools::fftd(xsf, 0, 1); // 假设 axis 1 表示按列 FFT（对每个特征向量）

    return xsf;
}

// // #include <omp.h>
// cv::Mat FDSSTTracker::get_scale_sample(const cv::Mat &image)
// {
//     cv::Mat xsf;   // output
//     int totalSize; // # of features
//     using namespace FFTTools;

//     int num_threads = 3;
//     omp_set_num_threads(num_threads);

//     #pragma omp parallel
//     {
//         float cx = _roi.x + _roi.width / 2.0f;
//         float cy = _roi.y + _roi.height / 2.0f;
//         #pragma omp for nowait
//         for (int i = 0; i < n_scales; i++) {
//             // Size of subwindow waiting to be detect
//             float patch_width = base_width * scaleFactors[i] * currentScaleFactor;
//             float patch_height = base_height * scaleFactors[i] * currentScaleFactor;

//             // Get the subwindow
//             cv::Mat im_patch = RectTools::extractImage(image, cx, cy, patch_width, patch_height);
//             cv::Mat im_patch_resized;

//             // Scaling the subwindow
//             if (scale_model_width > im_patch.cols)
//                 resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);
//             else
//                 resize(im_patch, im_patch_resized, cv::Size(scale_model_width, scale_model_height), 0, 0, cv::INTER_LINEAR);

//             cv::Mat hogs = fhog(im_patch_resized, cell_size);

//             if (i == 0)
//             {
//                 totalSize = hogs.cols * hogs.rows * 32;
//                 xsf = cv::Mat(cv::Size(n_scales, totalSize), CV_32F, float(0));
//             }

//             // Multiply the FHOG results by hanning window and copy to the output
//             cv::Mat FeaturesMap = hogs.reshape(1, totalSize);
//             float mul = s_hann.at<float>(0, i);
//             FeaturesMap = mul * FeaturesMap;
            
//             #pragma omp critical
//             FeaturesMap.copyTo(xsf.col(i));
//         }
//     }

//     // Do fft to the FHOG features row by row
//     xsf = FFTTools::fftd(xsf, 0, 1);

//     return xsf;
// }
#endif
// Compute the FFT Guassian Peak for scaling
cv::Mat FDSSTTracker::computeYsf()
{
    float scale_sigma2 = n_scales / std::sqrt(n_scales) * scale_sigma_factor;
    scale_sigma2 = scale_sigma2 * scale_sigma2;
    cv::Mat res(cv::Size(n_scales, 1), CV_32F, float(0));
    float ceilS = std::ceil(n_scales / 2.0f);

    for (int i = 0; i < n_scales; i++)
    {
        res.at<float>(0, i) = std::exp(-0.5 * std::pow(i + 1 - ceilS, 2) / scale_sigma2);
    }

    return FFTTools::fftd(res);
}

// Compute the hanning window for scaling
cv::Mat FDSSTTracker::createHanningMatsForScale()
{
    cv::Mat hann_s = cv::Mat(cv::Size(n_scales, 1), CV_32F, cv::Scalar(0));
    for (int i = 0; i < hann_s.cols; i++)
        hann_s.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann_s.cols - 1)));

    return hann_s;
}

cv::Mat FDSSTTracker::resizeDFT(const cv::Mat &A, int real_scales)
{
    float scaling = (float)real_scales / n_scales;

    cv::Mat M = cv::Mat(cv::Size(real_scales, 1), CV_32FC2, cv::Scalar(0));

    int mids = ceil(n_scales / 2);
    int mide = floor((n_scales - 1) / 2) - 1;

    A *= scaling;

    A(cv::Range::all(), cv::Range(0, mids)).copyTo(M(cv::Range::all(), cv::Range(0, mids)));

    A(cv::Range::all(), cv::Range(n_scales - mide - 1, n_scales)).copyTo(M(cv::Range::all(), cv::Range(real_scales - mide - 1, real_scales)));

    return M;
}

#endif