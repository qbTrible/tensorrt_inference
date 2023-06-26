CMAKE_MINIMUM_REQUIRED (VERSION 3.13 FATAL_ERROR)

# Enable the languages which in use
ENABLE_LANGUAGE (C CXX)

# C++11 is the base required standard
SET (CMAKE_CXX_STANDARD 11)
SET (CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET (CMAKE_CXX_EXTENSIONS OFF)

ADD_COMPILE_OPTIONS(-fPIC)
# SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Project name
PROJECT (TRT_INFER)

# cuda conversion选项
OPTION (USE_CCA  "Build with cuda conversion"  ON)
IF (USE_CCA)
    message(STATUS "Using CCA ${USE_CCA}")
    ADD_DEFINITIONS(-DUSE_CCA)
ENDIF ()

IF (NOT CMAKE_BUILD_TYPE)
    SET (CMAKE_BUILD_TYPE debug CACHE STRING "Choose the type of build" FORCE)
ENDIF()

# file(GLOB src ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB src test.cpp trt_inference.cpp)

# 添加编译选项
ADD_COMPILE_OPTIONS(-fPIC)

# opencv
SET (OpenCV_DIR /home/linux/project/quanbo/iaif/3rdparty/opencv)
FIND_PACKAGE(OpenCV REQUIRED)
IF (NOT OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV not found")
ENDIF ()

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBDIR})

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
# arch、code根据实际显卡算力填写
set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
# 允许CUDA语法
enable_language(CUDA)

# 根据.cu文件生成动态链接库文件
file(GLOB tensor_rt_cu ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
message(STATUS "cu file: " ${tensor_rt_cu})
add_library(cca SHARED ${tensor_rt_cu})

# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
# 动态链接库
link_directories(/usr/lib/aarch64-linux-gnu/)

# Target
ADD_EXECUTABLE(${PROJECT_NAME}
    ${src}
)
message(STATUS "cca: " cca)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer cudart nvinfer_plugin cca)

add_dependencies(${PROJECT_NAME} cca)
