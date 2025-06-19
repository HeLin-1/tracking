
#include "svp_yolov5.h"
#include "ss_mpi_sys.h"

svp_yolov5::svp_yolov5(/* args */)
{
    modelId = 11;
    isCpuProcess_ = true;
}

svp_yolov5::~svp_yolov5()
{
}

Result svp_yolov5::Init()
{
    // Result ret;
    Result ret = InitResource();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init acl resource failed");
    }
    ret = InitModel();
    if (ret != SUCCESS)
    {
        ERROR_LOG("Init model failed");
        return FAILED;
    }

    ret = modelProcess.CreateInputBuf();
    if (ret != SUCCESS)
    {
        ERROR_LOG("CreateInputBuf failed");
        return FAILED;
    }

    if (!isCpuProcess_)
    {
        ret = modelProcess.SetDetParas(modelId);
        if (ret != SUCCESS)
        {
            ERROR_LOG("SetDetParas failed");
            return FAILED;
        }
    }

    ret = modelProcess.CreateTaskBufAndWorkBuf();
    if (ret != SUCCESS)
    {
        ERROR_LOG("CreateTaskBufAndWorkBuf failed");
        return FAILED;
    }

    return ret;
}

Result svp_yolov5::InitResource()
{
    // ACL init
    const char *aclConfigPath = "acl.json";
    svp_acl_error ret = svp_acl_init(aclConfigPath);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl init failed");
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    ret = svp_acl_rt_set_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl open device %d failed", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success", deviceId_);

    // set no timeout
    ret = svp_acl_rt_set_op_wait_timeout(0);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl set op wait time failed");
        return FAILED;
    }
    INFO_LOG("set op wait time success");

    // create context (set current)
    ret = svp_acl_rt_create_context(&context_, deviceId_);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl create context failed");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = svp_acl_rt_create_stream(&stream_);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl create stream failed");
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    svp_acl_rt_run_mode runMode;
    ret = svp_acl_rt_get_run_mode(&runMode);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("acl get run mode failed");
        return FAILED;
    }
    if (runMode != SVP_ACL_DEVICE)
    {
        ERROR_LOG("acl run mode failed");
        return FAILED;
    }
    INFO_LOG("get run mode success");
    return SUCCESS;
}

Result svp_yolov5::InitModel()
{
    // const string omModelPath = "yolov5_cpu_coco_self.om";
    const string omModelPath = "yolov5_person_car.om";
    Result ret = modelProcess.LoadModelFromFileWithMem(omModelPath.c_str());
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute LoadModelFromFileWithMem failed");
        return FAILED;
    }

    printf("LoadModelFromFileWithMem sucess \n");

    ret = modelProcess.CreateDesc();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateDesc failed");
        return FAILED;
    }

    printf("CreateDesc sucess \n");

    ret = modelProcess.CreateOutput();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }

    return SUCCESS;
}

void svp_yolov5::DestroyResource()
{
    svp_acl_error ret;
    if (stream_ != nullptr)
    {
        ret = svp_acl_rt_destroy_stream(stream_);
        if (ret != SVP_ACL_SUCCESS)
        {
            ERROR_LOG("destroy stream failed");
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr)
    {
        ret = svp_acl_rt_destroy_context(context_);
        if (ret != SVP_ACL_SUCCESS)
        {
            ERROR_LOG("destroy context failed");
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = svp_acl_rt_reset_device(deviceId_);
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("reset device failed");
    }
    INFO_LOG("end to reset device is %d", deviceId_);

    ret = svp_acl_finalize();
    if (ret != SVP_ACL_SUCCESS)
    {
        ERROR_LOG("finalize acl failed");
    }
    INFO_LOG("end to finalize acl");
}

Result svp_yolov5::Detect(ot_video_frame_info &frame_info, svp_npu_rect_info_t *rect_info)
{
    Result ret;
    svp_acl_error err;
    svp_acl_data_buffer *data_buffer = 0;
    void *ori_data = 0;
    size_t ori_size, ori_stride;
    // std::string filePath = "dog_bike_car_yolov5.bin";

    size_t devSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims inputDims;

    data_buffer = svp_acl_mdl_get_dataset_buffer(modelProcess.input_, 0);
    ori_data = svp_acl_get_data_buffer_addr(data_buffer);
    ori_size = svp_acl_get_data_buffer_size(data_buffer);
    ori_stride = svp_acl_get_data_buffer_stride(data_buffer);

    td_void *virt_addr = TD_NULL;
    virt_addr = ss_mpi_sys_mmap(frame_info.video_frame.phys_addr[0], frame_info.video_frame.height * frame_info.video_frame.stride[0] * 3 / 2);

    // printf("ori_size = %d", ori_size);

    err = svp_acl_update_data_buffer(data_buffer, virt_addr, ori_size, ori_stride);
    if (err != SUCCESS)
    {
        ERROR_LOG("svp_acl_update_data_buffer failed");
        modelProcess.DestroyInput();
        return FAILED;
    }
    ret = modelProcess.Execute();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute inference failed");
        modelProcess.DestroyInput();
        return FAILED;
    }

    // modelProcess.DumpModelOutputResult();
    // modelProcess.OutputModelResult(modelId);
    // modelProcess.get_svp_rio(rect_info);
    modelProcess.get_svp_rio_cpu(rect_info, modelId);
    return SUCCESS;
}

Result svp_yolov5::Detect(ImageData srcImage, svp_npu_rect_info_t *rect_info)
{
    // INFO_LOG("start to process file:%s", testFile[index].c_str());

    Result ret;
    svp_acl_error err;
    svp_acl_data_buffer *data_buffer = 0;
    void *ori_data = 0;
    size_t ori_size, ori_stride;
    std::string filePath = "dog_bike_car_yolov5.bin";

    size_t devSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims inputDims;

    data_buffer = svp_acl_mdl_get_dataset_buffer(modelProcess.input_, 0);
    ori_data = svp_acl_get_data_buffer_addr(data_buffer);
    ori_size = svp_acl_get_data_buffer_size(data_buffer);
    ori_stride = svp_acl_get_data_buffer_stride(data_buffer);

#if 0
    modelProcess.GetInputStrideParam(0, devSize, stride, inputDims);
    size_t dataSize = modelProcess.GetInputDataSize(0);
    void *picDevBuffer = svp_utils::Utils::GetDeviceBufferOfFile(filePath.c_str(), inputDims, stride, dataSize);

    INFO_LOG("devSize = %d  stride = %d inputDims counts = %d ", devSize, stride, inputDims.dim_count);
    INFO_LOG("ori_size = %d  ori_stride = %d ", ori_size, ori_stride);

    err = svp_acl_update_data_buffer(data_buffer, picDevBuffer, ori_size, ori_stride);
#endif
    err = svp_acl_update_data_buffer(data_buffer, srcImage.data.get(), ori_size, ori_stride);
    if (err != SUCCESS)
    {
        ERROR_LOG("svp_acl_update_data_buffer failed");
        modelProcess.DestroyInput();
        return FAILED;
    }
    ret = modelProcess.Execute();
    if (ret != SUCCESS)
    {
        ERROR_LOG("execute inference failed");
        modelProcess.DestroyInput();
        return FAILED;
    }

    // modelProcess.DumpModelOutputResult();
    // modelProcess.OutputModelResult(modelId);
    modelProcess.get_svp_rio(rect_info);
    return SUCCESS;
}
Result svp_yolov5::close()
{
    // release model input buffer
    modelProcess.DestroyInput();
    return SUCCESS;
}
