# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
#


if(CV_VERSION_348)
    include_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/include
    ./)

    ## add host lib path
    link_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/lib)

    aux_source_directory(. SOURCE)


    #
    add_library(kcf SHARED ${SOURCE})
    target_link_libraries(kcf m dl pthread opencv_core opencv_imgproc opencv_imgcodecs)
else()

    set(CMAKE_BUILD_TYPE "Release") #Release Debug
    set(CMAKE_CXX_FLAGS " -fopenmp") 
if(ENABLE_CROSS_X)
    SET(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc-11")
    SET(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++-11")
    set(OpenCV_DIR /usr/local/opencv410_cross/lib/cmake/opencv4) # 交叉编译
else()
    set(OpenCV_DIR /usr/local/opencv410_install/lib/cmake/opencv4) # 编译

endif()
    find_package(OpenCV REQUIRED)
    message(STATUS ${OpenCV_INCLUDE_DIRS})
    message(STATUS ${OpenCV_LIBS})

    # SET(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc-11")
    # SET(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++-11")

    set(NE10_INC ${PROJECT_SOURCE_DIR}/third_lib/Ne10/inc)
    set(NE10_LIB ${PROJECT_SOURCE_DIR}/third_lib/Ne10/lib/libNE10.a)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    include_directories(PRIVATE ${NE10_INC})
    include_directories(${OpenCV_INCLUDE_DIRS})
    aux_source_directory(. SOURCE)

    # 
    add_library(kcf SHARED ${SOURCE})
    # target_link_libraries(kcf ${OpenCV_LIBS} )
    target_link_libraries(kcf
    ${NE10_LIB}
     m dl pthread opencv_core opencv_imgproc opencv_imgcodecs)


endif()

# set_target_properties(algo_yolov5 PROPERTIES LINK_INTERFACE_LIBRARIES "")
# set_target_properties(algo_yolov5 PROPERTIES INTERFACE_LINK_LIBRARIES "")

# install(TARGETS algo_yolov5 DESTINATION .)

# 设置包含目录，使顶层可以找到子模块头文件
target_include_directories(kcf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})



# # CMake lowest version requirement
# cmake_minimum_required(VERSION 3.5.1)
# #
# ## Header path
# include_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/include
#                     ./)

# ## add host lib path
# link_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/lib)

# aux_source_directory(. SOURCE)


# #
# add_library(kcf SHARED ${SOURCE})
# target_link_libraries(kcf m dl pthread opencv_core opencv_imgproc opencv_imgcodecs)

# # set_target_properties(algo_yolov5 PROPERTIES LINK_INTERFACE_LIBRARIES "")
# # set_target_properties(algo_yolov5 PROPERTIES INTERFACE_LINK_LIBRARIES "")

# # install(TARGETS algo_yolov5 DESTINATION .)