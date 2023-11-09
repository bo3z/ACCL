# /*******************************************************************************
#  Copyright (C) 2022 Xilinx, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# *******************************************************************************/

# Environment, C++ and CMake version
cmake_minimum_required(VERSION 3.16) 
project(mvm_test_accl)
set(CMAKE_CXX_STANDARD 17) 

# Base ACCL repo
set(ACCL_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../../../../)

# Set useful compile warnings
add_compile_options(-Wall -Wextra -Wno-unused-variable
                    -Wno-unused-but-set-variable -Wno-unused-parameter)

# Set default output directory
if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
endif()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
endif()

# Include paths
set(ACCL_HLSDRV_DIR ${ACCL_REPO_ROOT}/driver/hls/) # ACCL Drivers
set(HLSLIB_INCLUDE ${ACCL_REPO_ROOT}/hlslib/include/hlslib/xilinx/) # HLS Lib
set(CCLO_BFM_DIR ${ACCL_REPO_ROOT}/test/model/bfm/) # ACCL Bus-functional model for PL-PL emulation/simulation
set(MVM_INCLUDES ${ACCL_REPO_ROOT}/kernels/plugins/mvm/ ${ACCL_REPO_ROOT}/test/host/mvm/data) # Application kernel
set(EIGEN_INCLUDES ${ACCL_REPO_ROOT}/test/host/mvm/eigen) # Library for fast linear algebra
set(EMU_INCLUDES ${HLSLIB_INCLUDE} $ENV{XILINX_HLS}/include/ ${ACCL_HLSDRV_DIR} ${CCLO_BFM_DIR}) # Path for emulator includes

# Some environment flags
set(EN_AVX 1)
set(ACCL_DEBUG 1)
set(CMAKE_BUILD_TYPE Debug)
set(ACCL_NETWORK_UTILS_MPI 1)
set(ACCL_NETWORK_UTILS_DEBUG 1)

# ACCL directories that need to be present
add_subdirectory(${ACCL_REPO_ROOT}/driver/utils/accl_network_utils ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/accl_network_utils)
add_subdirectory(${ACCL_REPO_ROOT}/test/model/bfm/ ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/cclobfm)

# Find MPI package -- required
find_package(MPI REQUIRED)
message(STATUS "MPI Include Path: ${MPI_CXX_INCLUDE_PATH}")

# Add executables and binary output file
add_executable(test_accl test_accl.cpp ${CCLO_BFM_DIR}/cclo_bfm.cpp ${ACCL_REPO_ROOT}/kernels/plugins/mvm/mvm_kernel.cpp)

# Add require libraries
target_include_directories(test_accl PUBLIC ${MPI_CXX_INCLUDE_PATH} ${ACCL_INCLUDE_PATH} ${EMU_INCLUDES} ${MVM_INCLUDES} ${EIGEN_INCLUDES})
target_link_libraries(test_accl PUBLIC MPI::MPI_CXX zmq zmqpp pthread cclobfm accl_network_utils)

# Explicitely link against vnx and network_roce_v2 because CMake doesn't support nested object libraries
# https://stackoverflow.com/a/71040950
target_link_libraries(test_accl PUBLIC vnx network_roce_v2)

# Compile options & optimization
target_compile_options(test_accl PRIVATE -w -fdiagnostics-color=always -g -O3)
