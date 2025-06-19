# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
#
## Header path
SET(CV_VERSION_348 OFF)

if(CV_VERSION_348)
message(STATUS "Using OpenCV 3.4.8")
    include_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/include
    ./)

    ## add host lib path
    link_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/lib)

    aux_source_directory(. SOURCE)


    #
    add_library(lin_kcf SHARED ${SOURCE})
    target_link_libraries(lin_kcf m dl pthread opencv_core opencv_imgproc opencv_imgcodecs)
else()
message(STATUS "Using OpenCV 4.10.0")
    set(OpenCV_DIR /usr/local/opencv410_lib/lib/cmake/opencv4)

    find_package(OpenCV REQUIRED)
    message(STATUS ${OpenCV_INCLUDE_DIRS})
    message(STATUS ${OpenCV_LIBS})

    include_directories(${OpenCV_INCLUDE_DIRS})
    aux_source_directory(. SOURCE)

    # 
    add_library(lin_kcf SHARED ${SOURCE})
    target_link_libraries(lin_kcf ${OpenCV_LIBS} )

endif()

# set_target_properties(algo_yolov5 PROPERTIES LINK_INTERFACE_LIBRARIES "")
# set_target_properties(algo_yolov5 PROPERTIES INTERFACE_LINK_LIBRARIES "")

# install(TARGETS algo_yolov5 DESTINATION .)

# 设置包含目录，使顶层可以找到子模块头文件
target_include_directories(lin_kcf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
