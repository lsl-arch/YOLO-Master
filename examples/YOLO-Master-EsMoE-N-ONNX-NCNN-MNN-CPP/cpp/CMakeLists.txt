cmake_minimum_required(VERSION 3.16)
project(yolomaster_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# PORTABLE=ON -> slim relocatable bundle: link only the OpenCV modules headless
# image inference needs (drops videoio/highgui -> no FFmpeg/GStreamer/X11 closure).
option(PORTABLE "Slim relocatable build: image inference only, minimal OpenCV deps" OFF)
# image I/O via stb + hand-written NMS -> only need core+imgproc (no imgcodecs->GDAL,
# no dnn->protobuf). Full build adds videoio for cv::VideoCapture (--source video).
if(PORTABLE)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc)
else()
  find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)   # -DOpenCV_DIR=... on Windows
endif()

# SDK roots — override per platform:
#   Linux : cmake -DONNXRUNTIME_ROOT=/path -DNCNN_ROOT=/path ..
#   Windows: cmake -DONNXRUNTIME_ROOT=C:/dev/onnxruntime -DNCNN_ROOT=C:/dev/ncnn -DOpenCV_DIR=C:/dev/opencv/build ..
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNXRuntime root")
set(NCNN_ROOT        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn"        CACHE PATH "ncnn root")

option(USE_ORT  "Build ONNXRuntime backend" ON)
option(USE_NCNN "Build ncnn backend"        ON)
option(USE_CUDA "Enable ONNX Runtime CUDA execution provider (requires a CUDA-enabled ORT SDK)" OFF)
option(ALLOW_NO_BACKENDS
  "Allow a dependency-only CLI build when requested backend SDKs are absent (diagnostic only)"
  OFF)
if(USE_CUDA AND NOT USE_ORT)
  message(FATAL_ERROR "USE_CUDA=ON requires USE_ORT=ON")
endif()

set(SRC  src/common.cpp src/main.cpp src/stb_impl.cpp)
set(INCS include third_party ${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})
set(DEFS "")
if(NOT PORTABLE)
  list(APPEND DEFS HAVE_VIDEOIO)   # cv::VideoCapture (--source video) needs opencv_videoio
endif()
set(RPATH "")
set(RUNTIME_DLLS "")   # Windows: DLLs to copy next to the .exe

if(USE_ORT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  list(APPEND SRC  src/ort_backend.cpp)
  list(APPEND INCS ${ONNXRUNTIME_ROOT}/include)
  find_library(ORT_LIB NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib/x64" NO_DEFAULT_PATH)
  if(NOT ORT_LIB)
    file(GLOB ORT_VERSIONED_LIBS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.*")
    list(LENGTH ORT_VERSIONED_LIBS ORT_VERSIONED_COUNT)
    if(ORT_VERSIONED_COUNT GREATER 0)
      list(GET ORT_VERSIONED_LIBS 0 ORT_LIB)
    endif()
  endif()
  if(NOT ORT_LIB)
    message(FATAL_ERROR "ONNX Runtime headers found at ${ONNXRUNTIME_ROOT}, but the runtime library was not found")
  endif()
  if(USE_CUDA)
    file(READ "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h" ORT_CXX_API_TEXT)
    string(FIND "${ORT_CXX_API_TEXT}" "AppendExecutionProvider_CUDA" ORT_CUDA_API_POS)
    if(ORT_CUDA_API_POS EQUAL -1)
      message(FATAL_ERROR "USE_CUDA=ON, but this ONNX Runtime SDK does not expose AppendExecutionProvider_CUDA; install a CUDA-enabled ORT SDK or set -DUSE_CUDA=OFF")
    endif()
    list(APPEND DEFS USE_CUDA)
  endif()
  if(WIN32)
    list(APPEND LIBS ${ORT_LIB})
    file(GLOB ORT_DLLS
      "${ONNXRUNTIME_ROOT}/*.dll"
      "${ONNXRUNTIME_ROOT}/lib/*.dll"
      "${ONNXRUNTIME_ROOT}/bin/*.dll")
    list(APPEND RUNTIME_DLLS ${ORT_DLLS})
  else()
    list(APPEND LIBS ${ORT_LIB})
    get_filename_component(ORT_LIB_DIR "${ORT_LIB}" DIRECTORY)
    list(APPEND RPATH ${ORT_LIB_DIR})
  endif()
  list(APPEND DEFS USE_ORT)
  message(STATUS "ONNXRuntime backend: ON (${ONNXRUNTIME_ROOT})")
else()
  if(USE_ORT AND NOT ALLOW_NO_BACKENDS)
    message(FATAL_ERROR
      "ONNX Runtime backend requested, but the SDK was not found at ${ONNXRUNTIME_ROOT}. "
      "Set -DONNXRUNTIME_ROOT=... or use -DALLOW_NO_BACKENDS=ON only for a diagnostic build.")
  endif()
  message(WARNING "ONNXRuntime backend: OFF (not found at ${ONNXRUNTIME_ROOT}; diagnostic build)")
endif()
if(USE_CUDA AND NOT (USE_ORT AND EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h"))
  message(FATAL_ERROR "USE_CUDA=ON requires an ONNX Runtime SDK at ${ONNXRUNTIME_ROOT}")
endif()

if(USE_NCNN AND EXISTS "${NCNN_ROOT}/include/ncnn/net.h")
  list(APPEND SRC  src/ncnn_backend.cpp)
  list(APPEND INCS ${NCNN_ROOT}/include ${NCNN_ROOT}/include/ncnn)
  find_library(NCNN_LIB NAMES ncnn
    PATHS "${NCNN_ROOT}/lib" "${NCNN_ROOT}/lib/x64" NO_DEFAULT_PATH)
  if(NOT NCNN_LIB)
    file(GLOB NCNN_VERSIONED_LIBS "${NCNN_ROOT}/lib/libncnn.so.*")
    list(LENGTH NCNN_VERSIONED_LIBS NCNN_VERSIONED_COUNT)
    if(NCNN_VERSIONED_COUNT GREATER 0)
      list(GET NCNN_VERSIONED_LIBS 0 NCNN_LIB)
    endif()
  endif()
  if(NOT NCNN_LIB)
    message(FATAL_ERROR "ncnn headers found at ${NCNN_ROOT}, but libncnn was not found under ${NCNN_ROOT}/lib")
  endif()
  list(APPEND LIBS ${NCNN_LIB})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    list(APPEND LIBS OpenMP::OpenMP_CXX)
  endif()
  if(WIN32)
    file(GLOB NCNN_DLLS
      "${NCNN_ROOT}/*.dll"
      "${NCNN_ROOT}/bin/*.dll"
      "${NCNN_ROOT}/lib/*.dll")
    list(APPEND RUNTIME_DLLS ${NCNN_DLLS})
  else()
    list(APPEND RPATH ${NCNN_ROOT}/lib)
  endif()
  list(APPEND DEFS USE_NCNN)
  message(STATUS "ncnn backend: ON (${NCNN_ROOT})")
else()
  if(USE_NCNN AND NOT ALLOW_NO_BACKENDS)
    message(FATAL_ERROR
      "ncnn backend requested, but the SDK was not found at ${NCNN_ROOT}. "
      "Set -DNCNN_ROOT=... or use -DALLOW_NO_BACKENDS=ON only for a diagnostic build.")
  endif()
  message(WARNING "ncnn backend: OFF (not found at ${NCNN_ROOT}; diagnostic build)")
endif()

if(NOT USE_ORT AND NOT USE_NCNN AND NOT ALLOW_NO_BACKENDS)
  message(FATAL_ERROR
    "Both inference backends are disabled. Enable USE_ORT or USE_NCNN, or set "
    "-DALLOW_NO_BACKENDS=ON for a diagnostic CLI build.")
endif()

add_executable(yolomaster_edge ${SRC})
set_target_properties(yolomaster_edge PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(yolomaster_edge PRIVATE ${INCS})
target_link_libraries(yolomaster_edge PRIVATE ${LIBS})
target_compile_definitions(yolomaster_edge PRIVATE ${DEFS})
if(MSVC)
  target_compile_options(yolomaster_edge PRIVATE /W4)
else()
  target_compile_options(yolomaster_edge PRIVATE -Wall -Wextra -Wpedantic)
  if(MINGW)
    # main.cpp exposes wmain so Windows command-line paths are received as
    # UTF-16 before conversion to UTF-8. MinGW needs this CRT startup switch;
    # MSVC selects wmain automatically.
    target_link_options(yolomaster_edge PRIVATE -municode)
  endif()
endif()

if(WIN32)
  # bundle the MSVC runtime (VCRUNTIME140.dll etc.) so the exe needs no VC++ Redist on targets
  set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
  include(InstallRequiredSystemLibraries)
  list(APPEND RUNTIME_DLLS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
  # Copy backend DLLs next to the exe. OpenCV can be shipped as one
  # opencv_world DLL or as modular core/imgproc/videoio DLLs; cover both
  # layouts so a clean Windows target does not depend on PATH.
  # OpenCV_DIR may point at <opencv>/build, .../x64/vcNN/lib, or a package lib.
  file(GLOB OCV_DLLS CONFIGURE_DEPENDS
    "${OpenCV_DIR}/x64/*/bin/opencv_*.dll"
    "${OpenCV_DIR}/../bin/opencv_*.dll"
    "${OpenCV_DIR}/../../bin/opencv_*.dll"
    "${OpenCV_DIR}/bin/opencv_*.dll")
  list(FILTER OCV_DLLS INCLUDE REGEX "opencv_.*\\.dll$")
  list(FILTER OCV_DLLS EXCLUDE REGEX "[0-9]d\\.dll$")   # drop the ~60MB debug opencv_world*d.dll
  list(REMOVE_DUPLICATES OCV_DLLS)
  foreach(dll ${RUNTIME_DLLS} ${OCV_DLLS})
    add_custom_command(TARGET yolomaster_edge POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "$<TARGET_FILE_DIR:yolomaster_edge>"
      COMMENT "copying ${dll}")
  endforeach()
else()
  set_target_properties(yolomaster_edge PROPERTIES BUILD_RPATH "${RPATH}")
endif()
