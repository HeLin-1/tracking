cmake_minimum_required(VERSION 3.5.0)
project(cv_track VERSION 0.1.0 LANGUAGES C CXX)


add_compile_options(-fPIC)
# add_compile_options(-march=armv8-a+fp+simd+crypto+crc)
# add_compile_options(-funroll-loops -DNDEBUG -DBOOST_DISABLE_ASSERTS )
add_definitions("-Wall -g") # -Wall：启用 GCC/Clang 的“几乎所有警告”选项，可以帮助开发者发现潜在的问题。-g：生成调试信息（符号表），用于调试工具（如 gdb）。
set(CMAKE_BUILD_TYPE "Release") #Release Debug
set(CMAKE_CXX_FLAGS " -fopenmp")

set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++")
set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc")

# 添加子目录
add_subdirectory(Algorithm)
# set(LIBRARY_OUTPUT_DIRECTORY &{PROJECT_SOURCE_DIR}/lib)
#
# 设置一个变量
## Header path
SET(CV_VERSION_348 OFF)
set(ENABLE_CROSS_X OFF) #cmake option
if(ENABLE_CROSS_X)
    # 通过 add_definitions 将其传递给编译器
    add_definitions(-DENABLE_CROSS_X)
    message(STATUS "DENABLE_CROSS_X")
endif()
set(ENABLE_FFTW_X OFF) #fftw lib option
# 根据变量值进行条件判断
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
	message(STATUS "PROCESSOR x86_64")
	if(ENABLE_CROSS_X)
		message(STATUS "ENABLE_CROSS_X is enabled")
		message(STATUS "Configuring for AArch64")
		#使用aarch64指定的编译器
		# SET(CMAKE_C_COMPILER "aarch64-mix210-linux-gcc")#aarch64-mix210-linux-gcc#/usr/bin/aarch64-linux-gnu-gcc-11
		# SET(CMAKE_CXX_COMPILER "aarch64-mix210-linux-g++")
		SET(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc-11")
		SET(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++-11")
		#aarch64 库 路径
        if(ENABLE_FFTW_X)
            set(FFTW_PATH "/usr/local/fftw_aarch64")
        endif()
        set(OpenCV_DIR /usr/local/opencv410_cross/lib/cmake/opencv4) # 交叉编译
        
	else()
		message(STATUS "ENABLE_CROSS_X is disabled")
		message(STATUS "Configuring for x86_64")
        if(ENABLE_FFTW_X)
            set(FFTW_PATH "/usr/local/fftw")
        endif()
        set(OpenCV_DIR /usr/local/opencv410_install/lib/cmake/opencv4) # x64 电脑编译
	endif()
else(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
	message(STATUS "PROCESSOR AArch64")
	message(STATUS "Configuring for AArch64")
    #aarch64 库 路径
    if(ENABLE_FFTW_X)
        set(FFTW_PATH "/usr/local/fftw_aarch64")
    endif()
    set(OpenCV_DIR /usr/local/opencv410_lib/lib/cmake/opencv4)
endif()

#set(OpenCV_DIR /usr/local/opencv410_install/lib/cmake/opencv4) # x64 电脑编译
# find_package(OpenCV REQUIRED)
# message(STATUS ${OpenCV_INCLUDE_DIRS})
# message(STATUS ${OpenCV_LIBS})

find_package(Eigen3 REQUIRED)

#set(FFTW_PATH "/usr/local/fftw")
if(ENABLE_FFTW_X)
    message("FFTW_PATH " ${FFTW_PATH})
    include_directories(${FFTW_PATH}/include)
    link_directories(${FFTW_PATH}/lib)
endif()




# include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})

include_directories(
    ${CMAKE_SOURCE_DIR}/Algorithm/bytetrack/include
    )

# add_executable(${PROJECT_NAME} main1.cpp func.cpp func.h)

# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} kcf )

if(ENABLE_FFTW_X)
    target_link_libraries(${PROJECT_NAME} -lfftw3f) #
endif()

#target_link_libraries(${PROJECT_NAME} libfftw3f.so ${OpenCV_LIBS})

if(CV_VERSION_348)
    message(STATUS CV_VERSION_348)
    message(STATUS CV_VERSION_348)
    ## Header path
    include_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/include
                        ./)

    ## add host lib path
    link_directories(${PROJECT_SOURCE_DIR}/third_lib/opencv-3.4.8/lib)

    add_executable(${PROJECT_NAME} main1.cpp func.cpp func.h)

    target_link_libraries(${PROJECT_NAME} m dl pthread opencv_core opencv_imgproc opencv_imgcodecs kcf)
else()
    find_package(OpenCV REQUIRED)
    message(STATUS ${OpenCV_INCLUDE_DIRS})
    message(STATUS ${OpenCV_LIBS})

    include_directories(${OpenCV_INCLUDE_DIRS})
    aux_source_directory(. SOURCE)

    # 
    add_executable(${PROJECT_NAME} 
    main1.cpp func.cpp func.h

    Algorithm/bytetrack/src/BytekalmanFilter.cpp
    Algorithm/bytetrack/src/BYTETracker.cpp
    Algorithm/bytetrack/src/lapjv.cpp
    Algorithm/bytetrack/src/STrack.cpp
    Algorithm/bytetrack/src/utils.cpp
    
    ) 
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} kcf fdsst)

endif()

add_executable(zoom digitalZoom.cpp)


add_executable(test main2.cpp ) 
target_link_libraries(test ${OpenCV_LIBS} kcf fdsst)