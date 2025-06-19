/**
 * @file utils.h
 *
 * Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include "svp_npu/svp_acl.h"
#include "svp_npu/svp_acl_mdl.h"

#include <memory>
#include <sstream>
#include "ot_common_video.h"
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#ifdef _WIN32
#define S_ISREG(m) (((m) & 0170000) == (0100000))
#endif
typedef enum Result
{
    SUCCESS = 0,
    FAILED = 1
} Result;

struct ImageData
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t alignWidth = 0;
    uint32_t alignHeight = 0;
    uint32_t size = 0;
    std::shared_ptr<uint8_t> data;
};

struct Rect
{
    uint32_t ltX = 0;
    uint32_t ltY = 0;
    uint32_t rbX = 0;
    uint32_t rbY = 0;
};

struct BBOX
{
    Rect rect;
    uint32_t score;
    int id;
};

#define SVP_RECT_POINT_NUM 4
typedef struct
{
    td_u16 class_id;
    float score;
    ot_point point[SVP_RECT_POINT_NUM];
} svp_npu_rect_t;

#define SVP_RECT_NUM 64
typedef struct
{
    td_u16 num;
    svp_npu_rect_t rect[SVP_RECT_NUM];
} svp_npu_rect_info_t;

struct DetBox
{
    float x;
    float y;
    float width;
    float height;
    std::string name;
};
enum class Yolo
{
    YOLOV1 = 1,
    YOLOV2 = 2,
    YOLOV3 = 3,
    YOLOV4 = 4,
    YOLOV5 = 5,
    YOLOV7 = 7,
    YOLOV8 = 8,
    YOLOX = 10,
    YOLOV5_CPU = 11,
    YOLOV7_CPU = 12,
    YOLOV8_CPU = 13,
};

namespace svp_utils
{

    class Utils
    {
    public:
        /**
         * @brief create device buffer of file
         * @param [in] fileName: file name
         * @param [out] fileSize: size of file
         * @return device buffer of file
         */
        static void *GetDeviceBufferOfFile(const std::string &fileName, const svp_acl_mdl_io_dims &dims,
                                           size_t stride, size_t dataSize);

        /**
         * @brief create buffer of file
         * @param [in] fileName: file name
         * @param [out] fileSize: size of file
         * @return buffer of pic
         */
        static void *ReadBinFile(const std::string &fileName, uint32_t &fileSize);

        static Result ReadFloatFile(const std::string &fileName, std::vector<float> &detParas);

        static Result GetFileSize(const std::string &fileName, uint32_t &fileSize);

        static void *ReadBinFileWithStride(const std::string &fileName, const svp_acl_mdl_io_dims &dims,
                                           size_t stride, size_t dataSize);

        static void InitData(int8_t *data, size_t dataSize);
    };

}
#endif