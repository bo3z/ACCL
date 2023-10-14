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
#
*******************************************************************************/

#include <vector>
#include <iostream>

#include <mpi.h>
#include <accl.hpp>
#include <tclap/CmdLine.h>
#include <accl_network_utils.hpp>

// Include the bus-functional model for emulation/simulation of the CCLO without hardware
#include "cclo_bfm.h"

// Application-specific kernels
#include "pl_reduction.h"

struct options_t {
    int start_port;
    unsigned int count;
    unsigned int rxbuf_size;
    unsigned int segment_size;
    unsigned int num_rxbufmem;
    bool hardware;
    bool tcp;
    std::string xclbin;
};

options_t parse_options(int argc, char *argv[]) {
    try {
        TCLAP::CmdLine cmd("Test ACCL C++ Driver");
    
        TCLAP::ValueArg<uint16_t> start_port_arg(
            "s", "start-port", "Start of range of ports usable for sim", 
            false, 5500, "positive integer"
        );
        cmd.add(start_port_arg);
    
        TCLAP::ValueArg<uint32_t> count_arg(
            "c", "count", "How many element per buffer",
            false, 16, "positive integer"
        );
        cmd.add(count_arg);
        
        TCLAP::ValueArg<uint16_t> bufsize_arg(
            "b", "rxbuf-size", "How many KB per RX buffer", 
            false, 256, "positive integer"
        );
        cmd.add(bufsize_arg);

        TCLAP::ValueArg<uint16_t> num_rxbufmem_arg (
            "m", "num_rxbufmem", "Number of memory banks used for rxbuf", 
            false, 4, "positive integer"
        );
        cmd.add(num_rxbufmem_arg);
   
        TCLAP::SwitchArg hardware_arg(
            "f", "hardware", "enable hardware mode", 
            cmd, false
        );

        TCLAP::SwitchArg tcp_arg(
            "t", "tcp", "Use TCP hardware setup", 
            cmd, false
        );
    
        TCLAP::ValueArg<std::string> xclbin_arg(
            "x", "xclbin", "xclbin of accl driver if hardware mode is used", 
            false, "accl.xclbin", "file");
        cmd.add(xclbin_arg);

        options_t opts;
        opts.start_port = start_port_arg.getValue();
        opts.count = count_arg.getValue();
        opts.rxbuf_size = bufsize_arg.getValue() * 1024;
        opts.segment_size = bufsize_arg.getValue();
        opts.num_rxbufmem = num_rxbufmem_arg.getValue();
        opts.hardware = hardware_arg.getValue();
        opts.tcp = tcp_arg.getValue();
        
        std::cout << "Done parsing options..." << std::endl;
        std::cout << "count: " << opts.count << ", rxbuf_size: " << opts.rxbuf_size << ", num_rxbufmem: " << opts.num_rxbufmem << std::endl;
        return opts;
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
}

// Just a helper function printing out stuff happening on one of the ranks (nodes)
void helper_print(std::string sentence, int rank) {
    std::cout << sentence << " (rank: " << rank << ")" << std::endl;
}

void run(
    ACCL::ACCL &accl,
    int current_rank,
    int world_size,
    options_t options
) {
    helper_print("Starting PL reduction test...", current_rank);

    // Initialize a CCLO BFM (emulation/simulation only) and streams as needed
    hlslib::Stream<command_word> callreq, callack;
    hlslib::Stream<stream_word, 512> data_cclo2krnl, data_krnl2cclo;
    std::vector<unsigned int> dest = {0}; 
    CCLO_BFM cclo(options.start_port, current_rank, world_size, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
    cclo.run();

    // Barrier to make sure all ranks have finished setting up
    MPI_Barrier(MPI_COMM_WORLD);

    // Run application kernel
    pl_reduce<4>(   
        current_rank, 
        world_size,
        accl.get_communicator_addr(),
        accl.get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32}),
        callreq,
        callack,
        data_krnl2cclo, 
        data_cclo2krnl
    );

    // Completed
    cclo.stop();
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {    
    options_t options = parse_options(argc, argv);

    // Initialization
    int current_rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    accl_network_utils::acclDesign design = accl_network_utils::acclDesign::AXIS3x;
    std::vector<ACCL::rank_t> ranks;
    ranks = accl_network_utils::generate_ranks(
        true, 
        current_rank, 
        world_size, 
        options.start_port, 
        options.rxbuf_size
    );
    
    xrt::device device{};
    if (options.hardware) {
        std::cout << "Supported for hardware not implemented yet..." << std::endl;
        exit(1);
    }

    std::unique_ptr<ACCL::ACCL> accl = accl_network_utils::initialize_accl(
        ranks, 
        current_rank, 
        !options.hardware, 
        design, device, 
        options.xclbin, 
        16,
        options.rxbuf_size, 
        options.segment_size, 
        false
    );
    accl->set_timeout(1e6);

    // Run tests
    run(*accl, current_rank, world_size, options);

    // Done
    MPI_Finalize();
}
