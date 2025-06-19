/*
Author: Christian Bailer
Contact address: Christian.Bailer@dfki.de
Department Augmented Vision DFKI

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

#pragma once

//#include <cv.h>

#ifndef _OPENCV_FFTTOOLS_HPP_
#define _OPENCV_FFTTOOLS_HPP_
#endif

//NOTE: FFTW support is still shaky, disabled for now.
/*#ifdef USE_FFTW
#include <fftw3.h>
#endif*/


namespace FFTTools
{

#include <NE10.h>
#include <vector>
#include <omp.h>
void rearrange(cv::Mat &img);
void showMagnitudeSpectrum(const cv::Mat& complexImage, const std::string& windowName);

class FFT2DProcessor {
public:
    FFT2DProcessor(int rows_, int cols_) {
        // Ensure NE10 is initialized
        if (ne10_init() != NE10_OK) {
            std::cerr << "FFT2DProcessor failed to initialize NE10." << std::endl;
        }
        cfg_row = ne10_fft_alloc_c2c_float32(cols_);
        cfg_col = ne10_fft_alloc_c2c_float32(rows_);
        this->rows = rows_;
        this->cols = cols_;
        std::cout << "FFT2DProcessor" << std::endl;
    }

    ~FFT2DProcessor() {
        if (cfg_row) ne10_fft_destroy_c2c_float32(cfg_row);
        if (cfg_col) ne10_fft_destroy_c2c_float32(cfg_col);
        std::cout << "~FFT2DProcessor" << std::endl;
    }
    static void getStatus(int rows_, int cols_) {
        int min_edge = (rows_ < cols_) ? rows_ : cols_;
        std::cout << "rows = " << rows_ << " cols = " << cols_ << " min_edge = " << min_edge<< std::endl;
        if(min_edge == 22 || min_edge == 18 || min_edge == 14) {
            m_neon_fft = false;
        } else {
            m_neon_fft = true;
        }
    }
    cv::Mat process(const cv::Mat&, bool);
    cv::Mat process_cv(cv::Mat, bool, bool);

private:
    ne10_fft_cfg_float32_t cfg_row = nullptr;
    ne10_fft_cfg_float32_t cfg_col = nullptr;
    int rows, cols;
public:
    static bool m_neon_fft;
};

bool FFT2DProcessor::m_neon_fft = false;


// 2D FFT 实现
cv::Mat FFT2DProcessor::process(const cv::Mat& input, bool backwards = false) {
    // Get image dimensions
    if ((rows != input.rows) || (cols != input.cols)) {
        std::cout << "rows != input.rows  cols != input.col." << std::endl;
        throw std::runtime_error("rows != input.rows  cols != input.col.");
    }

    if (!cfg_row || !cfg_col) {
        std::cout << "rows = " << rows << " cols = " << cols << std::endl;
        std::cout << "Failed to initialize NE10" << std::endl;
        throw std::runtime_error("Failed to allocate FFT configuration.");
    }
    // 分配临时缓冲区
    std::vector<ne10_fft_cpx_float32_t> temp(rows * cols);
    std::vector<ne10_fft_cpx_float32_t> row_in(cols), row_out(cols);
    std::vector<ne10_fft_cpx_float32_t> col_in(rows), col_out(rows);

    // 输出 Mat，初始化为与输入相同尺寸和类型
    cv::Mat output(rows, cols, CV_32FC2);

    cv::Mat input_;
    
    if (input.isContinuous()) {
        input_ = input;
        // std::cout << "is Continuous" << std::endl;
    } else {
        input_ = input.clone();
        // std::cout << "not Continuous" << std::endl;
    }

    for (int i = 0; i < rows; i++) {
        if(backwards) {
            std::memcpy(row_in.data(), input_.ptr<cv::Vec2f>(i), cols * sizeof(ne10_fft_cpx_float32_t));
        } else {
            const float* input_ptr = input_.ptr<float>(i);
            for (int j = 0; j < cols; j++) { // 准备行输入（实数输入，虚部为 0）
                row_in[j].r = input_ptr[j];
                row_in[j].i = 0.0f;
            }
        }
        // 执行  行 FFT
        ne10_fft_c2c_1d_float32_neon(row_out.data(), row_in.data(), cfg_row, backwards);
        // 存储到临时缓冲区
        memcpy(temp.data() + i * cols, row_out.data(), cols * sizeof(ne10_fft_cpx_float32_t));
    }

    // 对每一列进行 1D FFT
    // 直接访问连续内存 
    ne10_fft_cpx_float32_t* output_data = 
        reinterpret_cast<ne10_fft_cpx_float32_t*>(output.data);

    for (int j = 0; j < cols; j++) {
        // 准备列输入
        for (int i = 0; i < rows; i++) {
            col_in[i] = temp[i * cols + j];
        }
        // 执行列 FFT
        ne10_fft_c2c_1d_float32_neon(col_out.data(), col_in.data(), cfg_col, backwards);
        // 存储到输出 Mat
        for (int i = 0; i < rows; ++i) {
            // output_data[i * cols + j] = col_out[i];
            memcpy(&output_data[i*cols + j], &col_out[i], sizeof(ne10_fft_cpx_float32_t));
        }
    }

    return output;
}

// 2D FFT cv 实现
cv::Mat FFT2DProcessor::process_cv(cv::Mat img, bool backwards = false, bool byRow = false) {
    if (img.channels() == 1) {
        cv::Mat planes[] = {cv::Mat_<float> (img), cv::Mat_<float>::zeros(img.size())};
        cv::merge(planes, 2, img);
    }
    if(byRow) {
        std::cout << "byRow" << std::endl;
        std::cout << "img.size = " << img.size() << std::endl;
        cv::dft(img, img, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
        std::cout << "dft img.size = " << img.size() << std::endl;
    } else
        cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0 );
    return img;
}

// 全局线程池（每个线程一个 FFT 处理器）
static std::vector<std::unique_ptr<FFT2DProcessor>> fft_pool;

void init_fft_pool(int rows, int cols, int max_threads = omp_get_max_threads()) {
    fft_pool.resize(max_threads); // 4
    FFT2DProcessor::getStatus(rows, cols);
    if(!FFT2DProcessor::m_neon_fft) {
    // if(1) {
        std::cout << "cv::dft init" << std::endl;
        for (int i = 0; i < max_threads; ++i) {
            fft_pool[i] = std::make_unique<FFT2DProcessor>(rows, cols);
        }
    } else {
        std::cout << "ne10 init" << std::endl;
        for (int i = 0; i < max_threads; ++i) {
            fft_pool[i] = std::make_unique<FFT2DProcessor>(rows, cols);
        }
    }
}

// 调整画面尺寸为原尺寸的一半
void showScaleImg(std::string str, const cv::Mat &frame, float scale_)
{
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, cv::Size(frame.cols / scale_, frame.rows / scale_));
    // 显示结果
    cv::imshow(str, resizedFrame);
}
// 计算并显示幅度谱
void showMagnitudeSpectrum(const cv::Mat& complexImage, const std::string& windowName) {
    // 分离实部和虚部
    cv::Mat planes[2];
    cv::split(complexImage, planes);
    
    // 计算幅度谱
    cv::Mat magnitudeImage;
    cv::magnitude(planes[0], planes[1], magnitudeImage);
    
    // 对数变换增强显示
    magnitudeImage += cv::Scalar::all(1);
    cv::log(magnitudeImage, magnitudeImage);
#if 0
    // 归一化
    cv::normalize(magnitudeImage, magnitudeImage, 0, 1, cv::NORM_MINMAX);
    // 频谱中心化
    rearrange(magnitudeImage);
    // 显示频谱
    showScaleImg(windowName, magnitudeImage, 0.1); // 灰度显示
#endif
    // 归一化
    cv::normalize(magnitudeImage, magnitudeImage, 0, 255, cv::NORM_MINMAX);
    magnitudeImage.convertTo(magnitudeImage, CV_8UC1);
    
    // 频谱中心化
    rearrange(magnitudeImage);
    
    cv::Mat colorMapImg;
    cv::applyColorMap(magnitudeImage, colorMapImg, cv::COLORMAP_JET);
    // 显示频谱
    showScaleImg(windowName, colorMapImg, 0.1); // 伪彩色图
}


#if 1
	cv::Mat fftd(cv::Mat img, bool backwards = false, bool byRow = false)
{
    // std::cout << "img.size(): " << img.size() << std::endl; // 24

    if (img.channels() == 1)
    {
        cv::Mat planes[] = {cv::Mat_<float> (img), cv::Mat_<float>::zeros(img.size())};
        // cv::Mat planes[] = {cv::Mat_<double> (img), cv::Mat_<double>::zeros(img.size())};
        cv::merge(planes, 2, img);
    }
    // std::cout << "img.type(): " << img.type() << std::endl; // 13
    if(byRow)
      cv::dft(img, img, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
    else
      cv::dft(img, img, backwards ? (cv::DFT_INVERSE | cv::DFT_SCALE) : 0 );
    // std::cout << "img_out.type(): " << img.type() << std::endl; // 13
    // std::cout << "backwards: " << backwards << std::endl; 
    return img;
}

#else

cv::Mat fftd(cv::Mat input, bool backwards = false, bool byRow = false) {
    // std::cout << "input.size() = " << input.size() << std::endl;
      if(!FFT2DProcessor::m_neon_fft) {
        // std::cout << "cv::dft" << std::endl;
        int thread_id = omp_get_thread_num();
        // std::cout << "thread_id = " << thread_id << std::endl;
        return fft_pool[thread_id]->process_cv(input, backwards, byRow);
    } else {
        if(byRow) {
            std::cout << "byRow" << std::endl;
            std::cout << "input.size = " << input.size() << std::endl;
            cv::dft(input, input, (cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT));
            std::cout << "dft input.size = " << input.size() << std::endl;
            return input;
        }
        int thread_id = omp_get_thread_num();
        // std::cout << "thread_id = " << thread_id << std::endl;
        return fft_pool[thread_id]->process(input, backwards);
    }
}

#endif

cv::Mat real(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[0];
}

cv::Mat imag(cv::Mat img)
{
    std::vector<cv::Mat> planes;
    cv::split(img, planes);
    return planes[1];
}

cv::Mat magnitude(cv::Mat img)
{
    cv::Mat res;
    std::vector<cv::Mat> planes;
    cv::split(img, planes); // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    if (planes.size() == 1) res = cv::abs(img);
    else if (planes.size() == 2) cv::magnitude(planes[0], planes[1], res); // planes[0] = magnitude
    else assert(0);
    return res;
}

cv::Mat complexMultiplication(cv::Mat a, cv::Mat b, bool conj = false)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

	if (conj)
		pb[1] *= -1.0;

    std::vector<cv::Mat> pres;
    pres.push_back(pa[0].mul(pb[0]) - pa[1].mul(pb[1]));
    pres.push_back(pa[0].mul(pb[1]) + pa[1].mul(pb[0]));

    cv::Mat res;
    cv::merge(pres, res);

    return res;
}

cv::Mat complexDivisionReal(cv::Mat a, cv::Mat b)
{
    std::vector<cv::Mat> pa;
    cv::split(a, pa);

    std::vector<cv::Mat> pres;

    cv::Mat divisor = 1. / b;

    pres.push_back(pa[0].mul(divisor));
    pres.push_back(pa[1].mul(divisor));

    cv::Mat res;
    cv::merge(pres, res);
    return res;
}

cv::Mat complexDivision(cv::Mat a, cv::Mat b)
{
    std::vector<cv::Mat> pa;
    std::vector<cv::Mat> pb;
    cv::split(a, pa);
    cv::split(b, pb);

    cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));

    std::vector<cv::Mat> pres;

    pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
    pres.push_back((pa[1].mul(pb[0]) + pa[0].mul(pb[1])).mul(divisor));

    cv::Mat res;
    cv::merge(pres, res);
    return res;
}

void rearrange(cv::Mat &img)
{
    // img = img(cv::Rect(0, 0, img.cols & -2, img.rows & -2));
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    cv::Mat q0(img, cv::Rect(0, 0, cx, cy)); // Top-Left - Create a ROI per quadrant
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy)); // Top-Right
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy)); // Bottom-Left
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

    cv::Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
/*
template < typename type>
cv::Mat fouriertransFull(const cv::Mat & in)
{
    return fftd(in);

    cv::Mat planes[] = {cv::Mat_<type > (in), cv::Mat_<type>::zeros(in.size())};
    cv::Mat t;
    assert(planes[0].depth() == planes[1].depth());
    assert(planes[0].size == planes[1].size);
    cv::merge(planes, 2, t);
    cv::dft(t, t);

    //cv::normalize(a, a, 0, 1, CV_MINMAX);
    //cv::normalize(t, t, 0, 1, CV_MINMAX);

    // cv::imshow("a",real(a));
    //  cv::imshow("b",real(t));
    // cv::waitKey(0);

    return t;
}*/

void normalizedLogTransform(cv::Mat &img)
{
    img = cv::abs(img);
    img += cv::Scalar::all(1);
    cv::log(img, img);
    // cv::normalize(img, img, 0, 1, CV_MINMAX);
}

typedef std::vector<cv::Mat> ComplexMats;

ComplexMats MultiChannelsDFT(const cv::Mat &img, int flags = 0)
{
	std::vector<cv::Mat> chls;
	std::vector<cv::Mat> out;
	cv::split(img, chls);
	out.resize(chls.size());
	for (int i = 0; i < chls.size(); i++)
	{
		cv::dft(chls[i], out[i], cv::DFT_COMPLEX_OUTPUT);
		//out[i] = (out[i]);
	}
	// cv::Mat out_m;
	// cv::merge(out, out_m);
	return out;
}

ComplexMats ComplexMatsMultiMat(const ComplexMats &A, cv::Mat b)
	{
		
		ComplexMats out;
		out.resize(A.size());
		for (int i = 0; i < A.size(); i++)
		{
			out[i] = complexMultiplication(b, A[i]);
		}
		return out;
	}


ComplexMats ComplexMatsMultiComplexMats(const ComplexMats &A, const ComplexMats &B)
{

	ComplexMats out;
	assert(A.size() == B.size());
	out.resize(A.size());
	for (int i = 0; i < A.size(); i++)
	{
		out[i] = complexMultiplication(A[i], B[i]);
	}
	return out;
}

     ComplexMats MCComplexConjMultiplication(const ComplexMats &A)
	{

		ComplexMats out;
		out.resize(A.size());
		for (int i = 0; i < A.size(); i++)
		{
			out[i] = (complexMultiplication(A[i], A[i], true));
		}
		//cv::Mat out_m;
		//cv::merge(out, out_m);
		return out;
	}

	cv::Mat MCMulti(cv::Mat a, cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		cv::split(a, pa);

		std::vector<cv::Mat> pres;

		pres.resize(pa.size());

		for (int i = 0; i < pa.size(); i++)
			pres[i] = pa[i].mul(b);
		cv::Mat res;
		cv::merge(pres, res);

		return res;
	}


	cv::Mat MCSum(const ComplexMats &a)
	{
		//std::vector<cv::Mat> pa;
		//cv::split(a, pa);
		assert(a.size() != 0);
		cv::Mat out;
		a[0].copyTo(out);
		for (int i = 1; i < a.size(); i++)
			out = out + a[i];

		return out;
	}

	cv::Mat MCSum(const cv::Mat &a)
	{
		std::vector<cv::Mat> pa;
		cv::split(a, pa);
		assert(pa.size() != 0);
		cv::Mat out;
		pa[0].copyTo(out);
		for (int i = 1; i < pa.size(); i++)
			out = out + pa[i];

		return out;
	}

}
