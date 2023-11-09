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
project(mvm_test_single)
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
set(MVM_INCLUDES ${ACCL_REPO_ROOT}/test/host/mvm/data)      # Application kernel
set(EIGEN_INCLUDES ${ACCL_REPO_ROOT}/test/host/mvm/eigen)   # Library for fast linear algebra

# Some environment flags
set(CMAKE_BUILD_TYPE Debug)

# Add executables and binary output file
add_executable(test_single test_single.cpp)

# Add require libraries
target_include_directories(test_single PUBLIC ${MVM_INCLUDES} ${EIGEN_INCLUDES})

# Compile options & optimization
target_compile_options(test_single PRIVATE -w -fdiagnostics-color=always -g -O3)
