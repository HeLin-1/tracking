/**
 * @file model_process.h
 *
 * Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MODEL_PROCESS_H
#define MODEL_PROCESS_H

#include <iostream>
#include <vector>
#include "svp_utils.h"
#include "svp_npu/svp_acl.h"
namespace svp_yolov5_model
{
    class ModelProcess
    {
    public:
        /**
         * @brief Constructor
         */
        ModelProcess();

        /**
         * @brief Destructor
         */
        ~ModelProcess();

        /**
         * @brief load model from file with mem
         * @param [in] modelPath: model path
         * @return result
         */
        Result LoadModelFromFileWithMem(const std::string &modelPath);

        /**
         * @brief unload model
         */
        void Unload();

        /**
         * @brief create dataset
         * @return result
         */
        Result InitInput();

        /**
         * @brief create model desc
         * @return result
         */
        Result CreateDesc();

        /**
         * @brief destroy desc
         */
        void DestroyDesc();

        /**
         * @brief create model input
         * @param [in] inputDataBuffer: input buffer
         * @param [in] bufferSize: input buffer size
         * @return result
         */
        Result CreateInput(void *inputDataBuffer, size_t bufferSize, int stride);

        void *GetpicDevBuffer(ImageData data, const svp_acl_mdl_io_dims &dims, size_t stride, size_t dataSize);

        Result CreateInputBuf();

        Result CreateTaskBufAndWorkBuf();

        Result SetDetParas(int32_t modelId);

        /**
         * @brief destroy input resource
         */
        void DestroyInput();

        /**
         * @brief create output buffer
         * @return result
         */
        Result CreateOutput();

        /**
         * @brief destroy output resource
         */
        void DestroyOutput();

        /**
         * @brief model execute
         * @return result
         */
        Result Execute();

        /**
         * @brief dump model output result to file
         */
        void DumpModelOutputResult() const;

        /**
         * @brief get model output result
         */
        void OutputModelResult(int32_t modelId) const;

        Result CreateBuf(int index);

        Result GetInputStrideParam(int index, size_t &bufSize, size_t &stride, svp_acl_mdl_io_dims &dims) const;

        Result GetOutputStrideParam(int index, size_t &bufSize, size_t &stride, svp_acl_mdl_io_dims &dims) const;

        size_t GetInputDataSize(int index) const;

        size_t GetOutputDataSize(int index) const;

    public:
        void WriteOutput(const std::string &outputFileName, size_t index) const;

        Result ClearOutputStrideInvalidBuf(std::vector<int8_t> &buffer, size_t index) const;

        Result SetDetParas(const std::vector<float> &detPara);
        void OutputModelResultYoloV(int32_t modelId) const;
        void OutputModelResultYoloVCpu(int32_t modelId) const;
        void OutputModelResultYoloV8Cpu(int32_t modelId) const;
        void FilterYolov5v7Box(int32_t modelId, std::vector<std::vector<float>> &vaildBox) const;
        void ProcessPerDectection(size_t detectIdx, std::vector<std::vector<float>> &vaildBox,
                                  std::vector<std::vector<uint32_t>> &anchorGrids) const;
        void FilterYolov8Box(int32_t modelId, std::vector<std::vector<float>> &vaildBox) const;

        uint32_t executeNum_{0};
        uint32_t modelId_{0};
        size_t modelMemSize_{0};
        size_t modelWeightSize_{0};
        void *modelMemPtr_{nullptr};
        void *modelWeightPtr_{nullptr};
        bool loadFlag_{false};
        svp_acl_mdl_desc *modelDesc_{nullptr};
        svp_acl_mdl_dataset *input_{nullptr};
        svp_acl_mdl_dataset *output_{nullptr};
        float scoreThr_{0.15};

        Result get_svp_rio(svp_npu_rect_info_t *rect_info);
        Result get_svp_rio_cpu(svp_npu_rect_info_t *rect_info, int32_t modelId);
    };
    constexpr uint8_t SCALE_SIZE = 3;
    // constexpr uint8_t CLASS_NUM = 80;
    // constexpr uint8_t OUT_PARM_NUM = 85; /* x, y, w,h, obj , class(80) */
    constexpr uint8_t CLASS_NUM = 2;
    constexpr uint8_t OUT_PARM_NUM = 7; /* x, y, w,h, obj , class(80) */

    struct DetectionInnerParam
    {
        float *outData{nullptr};
        size_t detectIdx{0};
        size_t wStrideOffset{0};
        float scoreThr{0.0f};
        uint32_t outWidth{0};
        uint32_t chnStep{0};
        uint32_t outHeightIdx{0};
        uint32_t objScoreOffset{0};
    };

}
#endif // MODEL_PROCESS_H