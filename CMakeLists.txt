cmake_minimum_required(VERSION 3.24)
set(CMAKE_POLICY_VERSION_MINIMUM "3.5.0" CACHE STRING "" FORCE)
project(SubQuadraticAudioTransformer LANGUAGES CXX)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization Flags
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/fp:fast> $<$<COMPILE_LANGUAGE:CXX>:/bigobj> $<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/MP>)
else()
    add_compile_options(-ffast-math)
endif()

# Build Options
option(USE_ROCM "Use AMD ROCm/HIP instead of NVIDIA CUDA" OFF)
option(FORCE_CPU "Force CPU fallback, disabling GPU support" OFF)
option(BUILD_TESTS "Build C++ unit tests" ON)
option(BUILD_CLI "Build standalone C++ CLI tool" ON)

include(CheckLanguage)

if(FORCE_CPU)
    message(STATUS "CPU Fallback explicitly enabled.")
    set(GPU_BACKEND "CPU")
else()
    if(USE_ROCM)
        check_language(HIP)
        if(CMAKE_HIP_COMPILER)
            enable_language(HIP)
            add_compile_definitions(USE_HIP)
            set(GPU_BACKEND "HIP")
            set(CMAKE_HIP_STANDARD 20)
            set(CMAKE_HIP_STANDARD_REQUIRED ON)
            set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -mllvm -amdgpu-early-inline-all=true")
        else()
            message(WARNING "HIP compiler not found. Falling back to CPU.")
            set(GPU_BACKEND "CPU")
        endif()
    else()
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            set(CMAKE_CUDA_ARCHITECTURES "native")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftz=true")
            add_compile_definitions(USE_CUDA)
            set(GPU_BACKEND "CUDA")
        else()
            check_language(HIP)
            if(CMAKE_HIP_COMPILER)
                enable_language(HIP)
                add_compile_definitions(USE_HIP)
                set(GPU_BACKEND "HIP")
                set(CMAKE_HIP_STANDARD 20)
                set(CMAKE_HIP_STANDARD_REQUIRED ON)
                set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3 -mllvm -amdgpu-early-inline-all=true")
            else()
                message(WARNING "Neither CUDA nor HIP compilers found. Falling back to CPU.")
                set(GPU_BACKEND "CPU")
            endif()
        endif()
    endif()
endif()

include(FetchContent)

# 1. Fetch pybind11
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use modern FindPython instead of legacy" FORCE)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# 2. Fetch nlohmann_json
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# 3. Fetch libsndfile
FetchContent_Declare(
  libsndfile
  GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
  GIT_TAG        1.2.2
)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libsndfile" FORCE)
set(BUILD_PROGRAMS OFF CACHE BOOL "Don't build libsndfile programs" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Don't build libsndfile examples" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Don't build libsndfile tests" FORCE)
set(ENABLE_CPU_CLIP OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libsndfile)

# 4. Fetch CLI11 (Command Line Argument Parser for C++20)
FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.2
)
FetchContent_MakeAvailable(cli11)

# 5. Fetch IXWebSocket (Live Network Data Ingestion)
set(USE_TLS OFF CACHE BOOL "Enable TLS/SSL support in IXWebSocket" FORCE)
set(USE_ZLIB OFF CACHE BOOL "Disable ZLIB in IXWebSocket for low latency raw PCM streaming" FORCE)
set(USE_WS OFF CACHE BOOL "Do not build IXWebSocket CLI tool" FORCE)

FetchContent_Declare(
  ixwebsocket
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG        v12.0.1
)
FetchContent_MakeAvailable(ixwebsocket)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define core library
set(SRC_FILES
    src/sha256.cpp
    src/audio_ingest.cpp
    src/network_ingest.cpp
    src/arena.cpp
    src/kernels/feature_map.cpp
    src/kernels/attention_forward.cpp
    src/kernels/attention_backward.cpp
    src/engine.cpp
    src/safetensors.cpp
    src/kernels/layer_norm.cpp
    src/kernels/ffn.cpp
)

if(GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(src/arena.cpp src/kernels/feature_map.cpp src/kernels/attention_forward.cpp src/kernels/attention_backward.cpp PROPERTIES LANGUAGE HIP)
elseif(GPU_BACKEND STREQUAL "CUDA")
    set_source_files_properties(src/arena.cpp src/kernels/feature_map.cpp src/kernels/attention_forward.cpp src/kernels/attention_backward.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(subq_core STATIC ${SRC_FILES})
target_link_libraries(subq_core PUBLIC 
    sndfile 
    nlohmann_json::nlohmann_json 
    ixwebsocket::ixwebsocket
)

if(WIN32)
    target_link_libraries(subq_core PUBLIC ws2_32 crypt32)
endif()

# Standalone Native C++ CLI Binary
if(BUILD_CLI)
    add_executable(subq_cli src/main.cpp)
    target_link_libraries(subq_cli PRIVATE 
        subq_core 
        CLI11::CLI11 
        sndfile 
        nlohmann_json::nlohmann_json
    )
    target_include_directories(subq_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

# Python Extension Module (_C)
pybind11_add_module(_C python/bindings.cpp)
target_link_libraries(_C PRIVATE subq_core)

if(MSVC)
    target_compile_options(_C PRIVATE /bigobj /fp:fast /EHsc /MP)
else()
    target_compile_options(_C PRIVATE -ffast-math -fPIC)
endif()

# Fallback output directory for direct CMake build when NOT building via scikit-build
if(NOT SKBUILD)
    set_target_properties(_C PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/subq_audio"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/subq_audio"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/subq_audio"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/subq_audio"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/subq_audio"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/subq_audio"
    )
endif()

install(TARGETS _C DESTINATION subq_audio)

# Unit Tests (Conditional)
if(BUILD_TESTS)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.5.3
    )
    FetchContent_MakeAvailable(Catch2)

    enable_testing()
    add_executable(test_arena tests/test_arena.cpp)
    target_link_libraries(test_arena PRIVATE subq_core Catch2::Catch2WithMain)

    add_executable(test_attention tests/test_attention.cpp)
    target_link_libraries(test_attention PRIVATE subq_core Catch2::Catch2WithMain)

    add_executable(test_safetensors tests/test_safetensors.cpp)
    target_link_libraries(test_safetensors PRIVATE subq_core Catch2::Catch2WithMain)

    add_executable(test_benchmark_audio tests/benchmark_audio.cpp)
    target_link_libraries(test_benchmark_audio PRIVATE subq_core Catch2::Catch2WithMain)

    add_executable(test_network_ingest tests/test_network_ingest.cpp)
    target_link_libraries(test_network_ingest PRIVATE subq_core Catch2::Catch2WithMain)

    include(CTest)
    include(Catch)
    catch_discover_tests(test_arena)
    catch_discover_tests(test_attention)
    catch_discover_tests(test_safetensors)
    catch_discover_tests(test_benchmark_audio)
    catch_discover_tests(test_network_ingest)
endif()
