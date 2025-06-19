
#ifndef _SVP_YOLOV5_H_
#define _SVP_YOLOV5_H_
#include "svp_utils.h"
#include "svp_npu/svp_acl.h"
#include <vector>
#include "model_process.h"

using namespace std;

class svp_yolov5
{
public:
    svp_yolov5(/* args */);
    ~svp_yolov5();
    Result InitResource();

    Result Init();

    Result InitModel();

    Result Detect(ImageData srcImage, svp_npu_rect_info_t *rect_info);

    Result Detect(ot_video_frame_info &frame_info, svp_npu_rect_info_t *rect_info);

    Result close();

    svp_yolov5_model::ModelProcess modelProcess;

private:
    /* data */
    void DestroyResource();
    int32_t modelId{0};
    int32_t deviceId_{0};
    svp_acl_rt_context context_{nullptr};
    svp_acl_rt_stream stream_{nullptr};
    bool isCpuProcess_{true};
};

#endif
