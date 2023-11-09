/*******************************************************************************
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
*******************************************************************************/
// iostream used purely for debugging purposes
// fstream is used for storing results to a text file, to be later compared with NumPy
#include <iostream>
#include <fstream>

// For accurate time measurements
#include <chrono>

// Library for fast linear algebra in C++
#include <Eigen/Dense>

// Contains the input vector and weight matrix
// See tesbtench.py on how its generated from Python
// In principle, it stores an N-dimensional vector and an MxN matrix
#include "mvm_data.h"

// Contains the number of inputs (N), number of outputs (M), world size (WS), root rank and data type (T)
// See tesbench.py on how its generated from Python
#include "mvm_constants.h"

int main(int argc, char *argv[]) {        
    T* result;
    
    // Begin application
    auto application_begin_time = std::chrono::high_resolution_clock::now();

    // Read weights and inputs for this rank
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix(M, N);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data(N, 1);
    for (int j = 0 ; j < M ; j++) {
        for (int k = 0 ; k < N ; k++) {
            matrix(j, k) = weights[j][k]; 
            data(k) = in[k];
        }
    }

    // Partial MVM, with data/weights for this rank
    auto computation_begin_time = std::chrono::high_resolution_clock::now();
    auto product = (matrix * data).eval();
    auto computation_end_time = std::chrono::high_resolution_clock::now();

    // Store to "local" memory
    result = product.data();
    auto application_end_time = std::chrono::high_resolution_clock::now();

    // Write outputs to text file, to be compared with target output later
    std::ofstream f("../log/single_log/single_outputs.txt");
    if (f.is_open()) {
        for (int i = 0; i < M; i++) {
            // Avoid extra space at the end, complicates parsing in Python
            std::string delim = (i == M - 1) ? "" : " "; 
            f << product(i, 0) << delim;
        }
        f.close();
    }

    // Print some timing information
    std::cout << "MVM: Computation duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(computation_end_time - computation_begin_time).count() << "ns" << std::endl;
    std::cout << "MVM: Results available on root duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(computation_end_time - computation_begin_time).count() << "ns" << std::endl;
    std::cout << "MVM: Application duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(application_end_time - application_begin_time).count() << "ns" << std::endl; 
}
