cmake_minimum_required(VERSION 3.10)

#Processing Parameters
if(NOT DEFINED LIBC_TYPE)
    set(LIBC_TYPE "uclibc")
endif()
if(DEFINED LIBC_TYPE)
    message(STATUS "Selected libc type: ${LIBC_TYPE}")
else()
    message(FATAL_ERROR "No libc type selected!")
    return()
endif()

set(SDK_PATH $ENV{LUCKFOX_SDK_PATH})
set(GLIBC_COMPILER $ENV{GLIBC_COMPILER})
if(${LIBC_TYPE} STREQUAL "uclibc")
    if (NOT SDK_PATH)
        message(WARNING "Please Set Luckfox-pico SDK Path. Such as:export LUCKFOX_SDK_PATH=/home/user/luckfox-pico")
        return()
    else()
        set(CMAKE_C_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-gcc")
        set(CMAKE_CXX_COMPILER "${SDK_PATH}/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++")
    endif()
elseif(${LIBC_TYPE} STREQUAL "glibc")
    if (NOT GLIBC_COMPILER)
        message(WARNING "Please Set arm32 glibc compiler Path. Such as:export GLIBC_COMPILER=/opt/arm-linux-gnueabihf/bin/arm-linux-gnueabihf-")
        return()
    else()
        set(CMAKE_C_COMPILER    "${GLIBC_COMPILER}gcc")
        set(CMAKE_CXX_COMPILER  "${GLIBC_COMPILER}g++")
    endif()
endif()

option(LUCKFOX_PICO_PLUS "Enable LUCKFOX_PICO_PLUS feature" OFF)
option(LUCKFOX_PICO_PRO_MAX "Enable LUCKFOX_PICO_PRO_MAX feature" OFF)

if(NOT DEFINED EXAMPLE_DIR)
    set(EXAMPLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/code")
endif()
message(STATUS "Selected source directory: ${EXAMPLE_DIR}")

if(NOT IS_DIRECTORY "${EXAMPLE_DIR}/src")
    message(FATAL_ERROR "Source directory not found: ${EXAMPLE_DIR}/src")
    return()
endif()

if(NOT IS_DIRECTORY "${EXAMPLE_DIR}/include")
    message(FATAL_ERROR "Include directory not found: ${EXAMPLE_DIR}/include")
    return()
endif()

if(NOT IS_DIRECTORY "${EXAMPLE_DIR}/model")
    message(FATAL_ERROR "Model directory not found: ${EXAMPLE_DIR}/model")
    return()
endif()

if(NOT DEFINED EXAMPLE_NAME)
    set(EXAMPLE_NAME "retinaface_facenet_spidev")
endif()
message(STATUS "Set project name: ${EXAMPLE_NAME}")

if(LUCKFOX_PICO_PLUS)
    add_definitions(-DLUCKFOX_PICO_PLUS)
    message(STATUS "Set Device: Luckfox Pico Plus")
endif()

if(LUCKFOX_PICO_PRO_MAX)
    add_definitions(-DLUCKFOX_PICO_PRO_MAX)
    message(STATUS "Set Device: Luckfox Pico Pro Max")
endif()

project(${EXAMPLE_NAME})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/${LIBC_TYPE})

add_definitions(-DRV1106_1103)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/Config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/GUI)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/LCD)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/SPI)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/GPIO)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/utils/ utils.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/Config)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/GUI)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/LCD)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/SPI)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/GPIO)

set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

#Opencv 4
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/${LIBC_TYPE}/lib/cmake/opencv4")
find_package(OpenCV REQUIRED QUIET)
if(NOT OpenCV_INCLUDE_DIRS AND IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/opencv2")
    set(OpenCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(COMMON_TARGET_LIBS
    ${OpenCV_LIBS}
    rknnmrt
    Threads::Threads
    imageutils
    fileutils
    imagedrawing
    Config
    GUI
    LCD
    SPI
    GPIO
)

if(${LIBC_TYPE} STREQUAL "glibc")
    list(APPEND COMMON_TARGET_LIBS
        stdc++
        func_none
        c
        m
        rt
        dl
    )
endif()

set(COMMON_TARGET_INCLUDES
    ${OpenCV_INCLUDE_DIRS}
    ${EXAMPLE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librga/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/allocator/dma
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Config
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/GUI
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/LCD
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/SPI
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/GPIO
)

set(SRC_DIR "${EXAMPLE_DIR}/src")
file(GLOB SRC_FILES_CC CONFIGURE_DEPENDS "${SRC_DIR}/*.cc")
file(GLOB SRC_FILES_C CONFIGURE_DEPENDS "${SRC_DIR}/*.c")
set(SRC_FILES ${SRC_FILES_CC} ${SRC_FILES_C})
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${COMMON_TARGET_LIBS})
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_TARGET_INCLUDES})

set(TOOLS_DIR "${EXAMPLE_DIR}/tools")

#Install
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install/RV1106_demo")

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES "${TOOLS_DIR}/face_db.txt" DESTINATION . OPTIONAL)
set(MODEL_DIR "${EXAMPLE_DIR}/model")
install(FILES
    "${MODEL_DIR}/facedet.rknn"
    "${MODEL_DIR}/facerecg.rknn"
    "${MODEL_DIR}/yolov5n.rknn"
    "${MODEL_DIR}/test.jpg"
    DESTINATION model
    OPTIONAL
)
install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/run_retinaface_facenet_spidev.sh" DESTINATION .)
install(CODE "file(MAKE_DIRECTORY \"${CMAKE_INSTALL_PREFIX}/faces\")")
