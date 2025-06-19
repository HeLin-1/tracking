# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
#
## Header path
include_directories(${CHIP_SDK}/include
)

## add host lib path
link_directories(${CHIP_SDK}/lib/svp_npu)

aux_source_directory(. SOURCE)

#list(REMOVE_ITEM SOURCE "./main.cpp")
#
add_library(svp_algo_yolov5 SHARED ${SOURCE})
target_link_libraries(svp_algo_yolov5 m dl pthread ${HISI_LIB})

# set_target_properties(algo_yolov5 PROPERTIES LINK_INTERFACE_LIBRARIES "")
# set_target_properties(algo_yolov5 PROPERTIES INTERFACE_LINK_LIBRARIES "")

# install(TARGETS algo_yolov5 DESTINATION .)
