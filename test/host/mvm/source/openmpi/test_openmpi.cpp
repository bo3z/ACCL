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
/////////////////////////////////////////////
//                INCLUDES                //
///////////////////////////////////////////
// iostream used purely for debugging purposes
// fstream is used for storing results to a text file, to be later compared with NumPy
#include <iostream>
#include <fstream>

// For accurate time measurements
#include <chrono>

// C++ libary for MPI environments and communications
#include <mpi.h>

// Library for fast linear algebra in C++
#include <Eigen/Dense>

// Contains the input vector and weight matrix
// See tesbtench.py on how its generated from Python
// In principle, it stores an N-dimensional vector and an MxN matrix
#include "mvm_data.h"

// Contains the number of inputs (N), number of outputs (M), world size (WS), root rank and data type (T)
// See tesbench.py on how its generated from Python
#include "mvm_constants.h"

/////////////////////////////////////////////
//             MAIN APPLICATION           //
///////////////////////////////////////////
// Important: https://eigen.tuxfamily.org/dox-3.2/TopicLazyEvaluation.html
// For lazy evaluation in Eigen

// Application code runs on the host CPU
void run_on_host_cluster(int current_rank) {
    // Buffer to store data to reduce
    T acc[M];
    T res[M];
    
    // Begin application
    auto application_begin_time = std::chrono::high_resolution_clock::now();

    // Read weights and inputs for this rank; initialize outputs
    // TODO - See if it can be further optimised
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix(M, N/WS);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data(N/WS, 1);
    for (int j = 0 ; j < M ; j++) {
        for (int k = 0 ; k < N/WS ; k++) {
            matrix(j, k) = weights[j][current_rank * (N/WS) + k]; 
            data(k) = in[current_rank * (N/WS) + k];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto computation_begin_time = std::chrono::high_resolution_clock::now();

    // Partial MVM, with data/weights for this rank
    auto product = (matrix * data).eval();
    auto computation_end_time = std::chrono::high_resolution_clock::now();

    // Store into buffers suitable for MPI REDUCE
    for (int i = 0; i < M; i++) {
        acc[i] = product(i, 0);
    }

    // Barrier to ensure everyone set up & reduce
    MPI_Reduce(acc, res, M, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    
    auto reduce_time = std::chrono::high_resolution_clock::now();
    auto application_end_time = std::chrono::high_resolution_clock::now();

    // Write outputs to text file, to be compared with target output later
    if (current_rank == root) {
        std::ofstream f("../log/openmpi_log/openmpi_outputs.txt");
        if (f.is_open()) {
            for (int i = 0; i < M; i++) {
                // Avoid extra space at the end, complicates parsing in Python
                std::string delim = (i == M - 1) ? "" : " "; 
                f << res[i] << delim;
            }
            f.close();
        } 
    }

    // Output some timing information
    if (root == current_rank) {
        std::cout << "MVM: Computation duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(computation_end_time - computation_begin_time).count() << "ns" << std::endl;
        std::cout << "MVM: Application duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(application_end_time - application_begin_time).count() << "ns" << std::endl;
        std::cout << "MVM: Results available on root duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(reduce_time - computation_begin_time).count() << "ns" << std::endl;
    }

    // Finish program
    MPI_Barrier(MPI_COMM_WORLD);
}

/////////////////////////////////////////////
//               MAIN                     //
///////////////////////////////////////////
int main(int argc, char *argv[]) {        
    // Initializes the MPI environment
    MPI_Init(&argc, &argv);

    // Returns the process ID (rank) of the processor (node) that called the function
    int current_rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    // Returns the total size of the environment via quantity of processes
    int world_size;     
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    assert(world_size == WS);
    
    // Barrier to make sure all ranks have been set up
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run application
    run_on_host_cluster(current_rank);

    // Cleans up the MPI environment and ends MPI communications
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
