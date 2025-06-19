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

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif


// 非线性 线性 高斯核
#define LINEAR_ 0

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 4;//2.5  // 1.2 - 3  -- 4
    // output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;
    // hog = true;
    //  guass_kernel = true;
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

        if (lab)
        {
            interp_factor = 0.005;
            sigma = 0.3;
            // output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(nClusters, 3, CV_32FC1, &data);
            cell_sizeQ = cell_size * cell_size;
        }
        else
        {
            _labfeatures = false;
        }
        std::cout << "sigma = " << sigma << std::endl;

    }
    else
    { // RAW
        interp_factor = 0.075;
        sigma = 0.2;
        cell_size = 1;
        _hogfeatures = false;

        if (lab)
        {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }

    if (multiscale)
    { // multiscale
        template_size = 96;
        // template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window)
        {
            // printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window)
    { // fit correction without multiscale
        template_size = 96;
        // template_size = 100;
        scale_step = 1;
    }
    else
    {
        template_size = 1;
        scale_step = 1;
    }
    std::cout << "KCFTracker" << std::endl;
}
KCFTracker::~KCFTracker()
{
    std::cout << "~KCFTracker" << std::endl;
}
// Initialize tracker
void KCFTracker::init(const cv::Rect &roi, cv::Mat image)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);
    _tmpl = getFeatures(image, 1);
    // std::cout << "_tmpl.size(): " << _tmpl.size() << std::endl; // 24
    std::cout << "size_patch[0]: " << size_patch[0] << std::endl;
    std::cout << "size_patch[1]: " << size_patch[1] << std::endl;
    // 在程序初始化时调用（确保线程池初始化）
    FFTTools::init_fft_pool(size_patch[0], size_patch[1]); // fft 尺寸 e.g 24x24

    _prob = createGaussianPeak(size_patch[0], size_patch[1]);
        // std::cout << "_prob.size(): " << _prob.size() << std::endl; // 24
    _alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame
}

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat image)
{
    cv::Rect i;
    return i;
}
// Update position based on the new frame
bool KCFTracker::update(cv::Mat image, cv::Rect &bbox, float kcf_threshold)
{

    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;

    float peak_value; // 峰值
    float apce_value;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value, apce_value);

    if (scale_step != 1)
    {
    	// Test at a smaller _scale
    	float new_peak_value;
    	cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value, apce_value);

    	if (scale_weight * new_peak_value > peak_value)
    	{
    		res = new_res;
    		peak_value = new_peak_value;
    		_scale /= scale_step;
    		_roi.width /= scale_step;
    		_roi.height /= scale_step;
    	}

    	// Test at a bigger _scale
    	new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value, apce_value);

    	if (scale_weight * new_peak_value > peak_value)
    	{
    		res = new_res;
    		peak_value = new_peak_value;
    		_scale *= scale_step;
    		_roi.width *= scale_step;
    		_roi.height *= scale_step;
    	}
    }

    // 2018/12/24 andyoyo add

    m_apceValue = apce_value;
    // 峰值小于设定阈值，认为跟踪失败
    printf("peak_value: %f\n", peak_value);
    if(padding>3) {
    if ((apce_value < 6 && peak_value < 0.4) || m_psr < 6.0) // 10 0.8
    {
        // bbox = _roi;
        return false;
    }
    } else 
    if ((apce_value < 12 && peak_value < 0.8) || m_psr < 8.0) // 10 0.8
    {
        // bbox = _roi;
        return false;
    }

    // static int failed_frames = 0;
    // if (peak_value < 0.3) {
    //     failed_frames++;
    //     if (failed_frames > 5) {
    //         std::cout << "连续失败，宣告丢失\n";
    //         return false;// 连续失败，宣告丢失
    //     }
    // } else {
    //     failed_frames = 0; // 重置计数器
    // }

    // 2018/12/24 andyoyo add
    //  Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1)
        _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1)
        _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0)
        _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0)
        _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);

    cv::Mat x = getFeatures(image, 0);
    float _factor = std::min(1.f, exp(4.f * (peak_value - 0.6f)));
    static double pre_psr = m_psr;
    if(m_psr < 15 || peak_value < 0.2) {
    // if(m_psr < 15 || peak_value < 0.2 || (pre_psr - m_psr > m_psr * 0.3 && m_psr < 25)) {
        _factor = 0;
        std::cout << ".........................不更新......\n";
    }
    pre_psr = m_psr;
    train(x, interp_factor * _factor);

    // train(x, interp_factor * std::min(1.f, exp(2.f * (peak_value - 1.f))));
    std::cout << "factor = " << _factor << std::endl;
    // bbox = _roi;// lin_edit
    bbox = cv::Rect(_roi) & cv::Rect(0, 0, image.cols, image.rows);
    return true;
}


// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value, float &apce_value)
{
    using namespace FFTTools;

    // cv::TickMeter time_ms, time_ms1;
    // time_ms1.start();
    // time_ms.start();
    cv::Mat k = gaussianCorrelation(x, z);
    // time_ms.stop();
    // std::cout << "gaussianCorrelation time = " << time_ms.getTimeMilli() << " ms" <<std::endl;
    // time_ms.reset();
    // time_ms.start();
#if LINEAR_
   cv::Mat res = (real(fftd(complexMultiplication(_alphaf, k), true))); //96 * 96
#else
   cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true))); //96 * 96
#endif
   
   // time_ms.stop();
    // std::cout << "complexMultiplication time = " << time_ms.getTimeMilli() << " ms" <<std::endl;
    // time_ms.reset();
    // Check if response map is valid
    if (res.cols < 3 || res.rows < 3) { // lin_edit
        std::cerr << "Response map too small!" << std::endl;
        peak_value = 0.0f;
        apce_value = 0.0f;
        return cv::Point2f(0, 0);
    }

    // minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;    
    //time_ms.start();

    cv::Mat show;
    cv::normalize(res, show, 0, 255, cv::NORM_MINMAX);
    show.convertTo(show, CV_8UC1);
    cv::Mat colorMapImg;
    cv::applyColorMap(show, colorMapImg, cv::COLORMAP_JET);
    FFTTools::showScaleImg("res", colorMapImg, 0.1);

    getAPCE(res, apce_value, pv, pi);
    peak_value = (float)pv;
    
    // // time_ms.stop();
    // std::cout << "time = " << time_ms.getTimeMilli() << " ms" <<std::endl;
    // time_ms.reset();

    // subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols - 1)
    {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
    }

    if (pi.y > 0 && pi.y < res.rows - 1)
    {
        p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;
    // time_ms1.stop();
    // std::cout << "detect time = " << time_ms1.getTimeMilli() << " ms" <<std::endl;
    // time_ms1.reset();

    return p;
}

// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);

#if LINEAR_
    cv::Mat alphaf = complexDivision(_prob, (k + lambda));
#else
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
#endif
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor)*x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor)*alphaf;

    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);

    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/
}


// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
/*
    原始代码
    高斯核
*/
#if !LINEAR_
    #define V2
#else
    #define V4
#endif

#ifdef V0
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat(cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0));
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++)
        {
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
    else
    {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d;
    cv::max(((cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0]) - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}
#endif
/*
    微调代码
    高斯核
*/
#ifdef V1
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    // std::cout << "x1.size() = " << x1.size() << std::endl; //576 * 31
    // std::cout << "x1 channels: " << x1.channels() << std::endl; // 1
    using namespace FFTTools;
    cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F);
    // size_patch[0] 24
    // size_patch[1] 24
    // size_patch[2] 31 
    if (_hogfeatures) {
        cv::Mat caux = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2); // 复数类型，预分配
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) { // 31
            x1aux = x1.row(i).reshape(1, size_patch[0]); // Procedure do deal with cv::Mat multichannel bug
            x2aux = x2.row(i).reshape(1, size_patch[0]); // 24 * 24
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            caux = fftd(caux, true);
            rearrange(caux);
            // caux.convertTo(caux, CV_32F);
            c = c + real(caux);
            // std::cout << "caux: " << caux.size() << "type: " << caux.type() << "channels: " << caux.channels() << std::endl;
        }
    } else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }

    cv::Mat d;
    double x1_energy = cv::sum(x1.mul(x1))[0]; // 可预计算
    double x2_energy = cv::sum(x2.mul(x2))[0]; // 可预计算
    cv::max((x1_energy + x2_energy - 2. * c) / (size_patch[0] * size_patch[1] * size_patch[2]), 0, d);

    cv::Mat k;
    cv::exp(-d / (sigma * sigma), k);
    return k;
}
#endif

/*
    并行代码
    高斯核
*/
#ifdef V2
#include <omp.h>
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F);
    if (_hogfeatures) {
        int num_threads = 3; // omp_get_max_threads(); 
#if HUIZONG
        std::vector<cv::Mat> c_locals(num_threads, cv::Mat::zeros(size_patch[0], size_patch[1], CV_32F));
#endif
    
        // 设置 OpenMP 线程数
        omp_set_num_threads(num_threads);
        // std::cout << "max_threads = " << omp_get_max_threads() << std::endl;
    
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
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

#ifdef V4
#include <omp.h>
// cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
// {
//     // std::cout << "x1.size() = " << x1.size() << std::endl; //576 *31
//     // std::cout << "x1 channels: " << x1.channels() << std::endl; // 1
//     using namespace FFTTools;
//     cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2);
//     int num_threads = 3; // omp_get_max_threads(); 
//     std::vector<cv::Mat> c_locals(num_threads, cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2));
//     // 设置 OpenMP 线程数
//     omp_set_num_threads(num_threads);
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         cv::Mat caux(size_patch[0], size_patch[1], CV_32FC2); // 复数矩阵
//         cv::Mat x1aux, x2aux, local_sum = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2); // 线程局部变量 预分配
//         // #pragma omp for nowait
//         // #pragma omp for schedule(dynamic, 1) 
//         #pragma omp for schedule(guided, 1) nowait
//         for (int i = 0; i < size_patch[2]; i++) {
//             x1aux = x1.row(i).reshape(1, size_patch[0]).clone();
//             x2aux = x2.row(i).reshape(1, size_patch[0]).clone();
//             cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
//             local_sum += caux;
//         }
//         // 线程结束后，汇总到 c_locals
//         c_locals[thread_id] = local_sum;
//     }

//     // 汇总所有线程的计算结果
//     for (const auto& c_local : c_locals) {
//         cv::add(c, c_local, c); // 避免直接 +=
//     }
//     float numel_xf = (size_patch[0] * size_patch[1] * size_patch[2]);

//     return c / numel_xf;
//     // cv::Mat k;
//     // k = c / numel_xf;
//     // return k;
// }

cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2) {
    using namespace FFTTools;
    cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2);
    int num_threads = 3;
    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        cv::Mat caux(size_patch[0], size_patch[1], CV_32FC2);
        cv::Mat x1aux, x2aux, local_sum = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2);

        #pragma omp for schedule(guided, 1) nowait
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i).reshape(1, size_patch[0]).clone();
            x2aux = x2.row(i).reshape(1, size_patch[0]).clone();
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
            local_sum += caux;
        }

        // 使用原子操作或临界区汇总结果
        #pragma omp critical
        c += local_sum;
    }

    float numel_xf = (size_patch[0] * size_patch[1] * size_patch[2]);
    return c / numel_xf;
}

// cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
// {
//     using namespace FFTTools;
//     cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2);
//     int num_threads = 3;
//     std::vector<cv::Mat> c_locals(num_threads, cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2));

//     omp_set_num_threads(num_threads);
//     #pragma omp parallel
//     {
//         int thread_id = omp_get_thread_num();
//         cv::Mat caux(size_patch[0], size_patch[1], CV_32FC2);
//         cv::Mat x1aux, x2aux;
//         cv::Mat& local_sum = c_locals[thread_id];
//         local_sum.setTo(cv::Scalar(0));  // 清零已有预分配内存

//         #pragma omp for schedule(dynamic, 1)
//         for (int i = 0; i < size_patch[2]; ++i) {
//             x1aux = x1.row(i).reshape(1, size_patch[0]);
//             x2aux = x2.row(i).reshape(1, size_patch[0]);
//             cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
//             local_sum += caux;
//         }
//     }

//     // 汇总阶段
//     for (int i = 1; i < num_threads; ++i) {
//         c_locals[0] += c_locals[i];
//     }

//     float numel_xf = (size_patch[0] * size_patch[1] * size_patch[2]);
//     return c_locals[0] / numel_xf;
// }

#endif

#ifdef V5
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    // std::cout << "x1.size() = " << x1.size() << std::endl; //576 *31
    // std::cout << "x1 channels: " << x1.channels() << std::endl; // 1
    using namespace FFTTools;
    cv::Mat c = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2);
    // size_patch[0] 24
    // size_patch[1] 24
    // size_patch[2] 31 
    cv::Mat caux = cv::Mat::zeros(size_patch[0], size_patch[1], CV_32FC2); // 复数类型，预分配
    cv::Mat x1aux;
    cv::Mat x2aux;
    for (int i = 0; i < size_patch[2]; i++) { // 31
        x1aux = x1.row(i).reshape(1, size_patch[0]); // Procedure do deal with cv::Mat multichannel bug
        x2aux = x2.row(i).reshape(1, size_patch[0]); // 24 * 24
        cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true);
        c = c + caux;
        // std::cout << "caux: " << caux.size() << "type: " << caux.type() << "channels: " << caux.channels() << std::endl;
    }

    float numel_xf = (size_patch[0] * size_patch[1] * size_patch[2]);
    cv::Mat k;
    k = c / numel_xf;
    return k;
}
#endif


// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
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
cv::Mat KCFTracker::getFeatures(const cv::Mat &image, bool inithann, float scale_adjust)
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
            _tmpl_sz.height = padded_h / _scale; // 96 * 96
        }
        else
        { // No template size given, use ROI size
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

        if (_hogfeatures)
        {
            // Round to cell size and also make it even
            _tmpl_sz.width = (((int)(_tmpl_sz.width / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
            _tmpl_sz.height = (((int)(_tmpl_sz.height / (2 * cell_size))) * 2 * cell_size) + cell_size * 2;
        }
        else
        { // Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }
    if(m_padding > 1e-6) {
        if(fabs(padding - m_padding) > 1e-6) {
            padding = m_padding;

            int padded_w = _roi.width * padding;
            int padded_h = _roi.height * padding;

            if (padded_w >= padded_h) // fit to width
                _scale = padded_w / (float)template_size;
            else
                _scale = padded_h / (float)template_size;
        } else {
            std::cout << "Padding = " << padding 
            << ", _scale = " << _scale 
            << std::endl;
        }
    }
    std::cout << "padding = " << padding 
    << ", _scale = " << _scale 
    << std::endl;   

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;
    std::cout << "extracted_roi = " << extracted_roi.size() << std::endl;
    cv::Mat FeaturesMap;
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
    std::cout << "z = " << z.size() << std::endl;

    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height)
    {
        cv::resize(z, z, _tmpl_sz);
    }
    std::cout << "z1 = " << z.size() << std::endl; // 104*104
    // HOG features
    if (_hogfeatures)
    {
        //IplImage z_ipl = z;
        IplImage z_ipl = cvIplImage(z); //lin_edit
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, cell_size, &map);
        normalizeAndTruncate(map, 0.2f);
        PCAFeatureMaps(map);
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;
        // std::cout << "    size_patch[0] = " << size_patch[0] << std::endl;
        // std::cout << "    size_patch[1] = " << size_patch[1] << std::endl;

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX * map->sizeY), CV_32F, map->map); // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);

        // Lab features
        if (_labfeatures)
        {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char *)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));

            int cntCell = 0;
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size)
            {
                for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size)
                {
                    // Iterate through each pixel of cell (cX,cY)
                    for (int y = cY; y < cY + cell_size; ++y)
                    {
                        for (int x = cX; x < cX + cell_size; ++x)
                        {
                            // Lab components for each pixel
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float *)(_labCentroids.data);
                            for (int k = 0; k < _labCentroids.rows; ++k)
                            {
                                float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k])) + ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1])) + ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
                                if (dist < minDist)
                                {
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ;
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and add features to FeaturesMap
            size_patch[2] += _labCentroids.rows;
            FeaturesMap.push_back(outputLab);
        }
    }
    else
    {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float)0.5; // In Paper;
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;
    }

    if (inithann)
    {
        createHanningMats();
    }
    FeaturesMap = hann.mul(FeaturesMap);
    return FeaturesMap;
}

// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{
    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1], 1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1, size_patch[0]), CV_32F, cv::Scalar(0));

    for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures)
    {
        cv::Mat hann1d = hann2d.reshape(1, 1); // Procedure do deal with cv::Mat multichannel bug

        hann = cv::Mat(cv::Size(size_patch[0] * size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++)
        {
            for (int j = 0; j < size_patch[0] * size_patch[1]; j++)
            {
                hann.at<float>(i, j) = hann1d.at<float>(0, j);
            }
        }
    }
    // Gray features
    else
    {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{
    float divisor = 2 * center - right - left;

    // if (divisor == 0)
    //     return 0;
    if (std::abs(divisor) < 1e-6) return 0.0f; // lin_edit

    return 0.5 * (right - left) / divisor;
}
#if 0
void KCFTracker::getAPCE1(const cv::Mat &im, float &apce)
{

    double maxVal = 0;
    cv::Point maxLoc;

    double minVal = 0;
    cv::Point minLoc;

    cv::minMaxLoc(im, &minVal, &maxVal, &minLoc, &maxLoc);

    double ave_res = 0;
    double value_diff = 0;
    double sum_diff = 0;

    double mmdiff = 0;
    mmdiff = maxVal - minVal;

    for (int i = 0; i < im.cols; i++)
    {
        for (int j = 0; j < im.rows; j++)
        {

            value_diff = im.at<float>(i, j) - minVal;
            if (value_diff > 0.0001 && value_diff < (mmdiff + 0.1))
                sum_diff += std::pow(value_diff, 2);
        }
    }
    ave_res = sum_diff / (im.cols * im.rows);

    mmdiff = std::pow(mmdiff, 2);
    apce = mmdiff / ave_res;

    std::cout << "apce = " << std::setprecision(10) << std::setw(15) << apce << std::endl;

    // std::cout << "最大值是：" << maxVal << std::endl;
    // std::cout << "灰度均值是：" << m << std::endl;
    // std::cout << "标准差是：" << s << std::endl;
    // double psr_data = (maxVal - m)/s;
    // std::cout << "psr = " << psr_data << std::endl;
}
#endif

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
void KCFTracker::getAPCE(const cv::Mat &im, float &apce, double &maxVal, cv::Point &maxLoc)
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


/*
    double minVal;
    cv::Point minLoc;
    cv::minMaxLoc(im, &minVal, &maxVal, &minLoc, &maxLoc);

    double mmdiff = maxVal - minVal;
    cv::Mat diff = im - minVal;
    cv::Mat diff_sq;
    cv::pow(diff, 2, diff_sq);
    double sum_diff = cv::sum(diff_sq)[0];
    double ave_res = sum_diff / (im.cols * im.rows);

    if (ave_res < 1e-6) {
        apce = 0.0f;
        return;
    }

    apce = std::pow(mmdiff, 2) / ave_res;

*/