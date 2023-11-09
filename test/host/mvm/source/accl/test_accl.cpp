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
// Vector is used to generate a list of ACCL ranks
#include <vector>

// iostream used purely for debugging purposes
// fstream is used for storing results to a text file, to be later compared with NumPy
#include <iostream>
#include <fstream>

// For parsing CLI arguments
#include <tclap/CmdLine.h>

// C++ libary for MPI environments and communications
#include <mpi.h>

// ACCL includes
#include <accl.hpp>
#include <accl_network_utils.hpp>

// Library for fast linear algebra in C++
#include <Eigen/Dense>

// Include the bus-functional model for emulation/simulation of the CCLO without hardware
// For more details, see: https://accl.readthedocs.io/en/latest/hls/simulation.html 
#include "cclo_bfm.h"

// Contains definitions for internet operations
#include <arpa/inet.h>

// Application-specific kernels/code
#include "mvm_kernel.h"

// Contains the input vector and weight matrix
// See tesbtench.py on how its generated from Python
// In principle, it stores an N-dimensional vector and an MxN matrix
#include "mvm_data.h"

// Contains the number of inputs (N), number of outputs (M), world size (WS), root rank and data type (T)
// See tesbench.py on how its generated from Python
#include "mvm_constants.h"

/////////////////////////////////////////////
//              HELPERS                   //
///////////////////////////////////////////
inline void _swap_endianness(uint32_t *ip) {
	uint8_t *ip_bytes = reinterpret_cast<uint8_t *>(ip);
	*ip = (ip_bytes[3] << 0) | (ip_bytes[2] << 8) | (ip_bytes[1] << 16) |
		  (ip_bytes[0] << 24);
}

inline uint32_t _ip_encode(std::string ip) {
	struct sockaddr_in sa;
	inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr));
	_swap_endianness(&sa.sin_addr.s_addr);
	return sa.sin_addr.s_addr;
}

/////////////////////////////////////////////
//             CLI OPTIONS                //
///////////////////////////////////////////
// A set of configurable options, varying performance, commnuication protocol, setup etc.
struct options_t {
    // Start of the port ranges 
	unsigned int start_port;

    // How many KB per RX Buffer - Typically better to keep it low in simulation
	unsigned int rxbuf_size;
    
    // Communication protocol
	bool udp, tcp, rdma;

    // Whether the application is executed as an FPGA kernel or on the host CPU
    bool kernel;

    // Whether the application is simulated or ran on hardware
    bool hardware;

    // Text file with FPGA IP addresses used in the application
    std::string fpgaIP;
};

options_t parse_options(int argc, char *argv[]) {
    try {
        TCLAP::CmdLine cmd("ACCL MVM C++ Driver");
    
        // Start port
        TCLAP::ValueArg<uint16_t> start_port_arg(
            "s", "start-port", "Start of range of ports usable for sim", 
            false, 5500, "Positive integer"
        );
        cmd.add(start_port_arg);

        // RX Buffer Size    
        TCLAP::ValueArg<uint16_t> bufsize_arg(
            "b", "rxbuf-size", "How many KB per RX buffer", 
            false, 4096, "Positive integer"
        );
        cmd.add(bufsize_arg);
   
        // Communication protocol
        TCLAP::SwitchArg tcp_arg(
            "t", "tcp", "Use TCP/IP communication", cmd, false
        );
        TCLAP::SwitchArg udp_arg(
            "u", "udp", "Use UDP communication", cmd, false
        );
        TCLAP::SwitchArg rdma_arg(
            "r", "rdma", "Use RDMA communication", cmd, false
        );

        // Where is the application executed?
        TCLAP::SwitchArg kernel_arg(
            "k", "kernel", "Run kernel on FPGA (1) or host (0)", cmd, false
        );

        // Running on hardware or not?
        TCLAP::SwitchArg hardware_arg(
            "f", "hardware", "Enable hardware mode", cmd, false
        );

        TCLAP::ValueArg<std::string> fpgaIP_arg(
			"l", "ipList", "ip list of FPGAs if hardware mode is used", 
            false, "fpga", "File name"
        );
        cmd.add(fpgaIP_arg);

        // Parse options
        std::cout << "Parsing options..." << std::endl;
        cmd.parse(argc, argv);

        // Sanity check - can't have more than one communication protocol
        if (
            udp_arg.getValue() + tcp_arg.getValue() + rdma_arg.getValue() != 1
        ) {
            throw std::runtime_error(
                "Specify exactly one communication protocol"
            );
        } 

        // Store to struct
        options_t opts;
        opts.start_port = start_port_arg.getValue();
        opts.rxbuf_size = bufsize_arg.getValue() * 1024; // Convert to bytes
        opts.tcp = tcp_arg.getValue();
        opts.udp = udp_arg.getValue();
        opts.rdma = rdma_arg.getValue();
        opts.hardware = hardware_arg.getValue();
        opts.kernel = kernel_arg.getValue();
		opts.fpgaIP = fpgaIP_arg.getValue();

        // Return to main application
        std::cout << "Done parsing options..." << std::endl;
        return opts;
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }

    MPI_Finalize();
    exit(1);
}

/////////////////////////////////////////////
//             MAIN APPLICATION           //
///////////////////////////////////////////
// Important: https://eigen.tuxfamily.org/dox-3.2/TopicLazyEvaluation.html
// For lazy evaluation in Eigen

// Application code runs on the host CPU
void run_on_host_cluster(ACCL::ACCL &accl, int current_rank, options_t options) {
    // Set up memory buffers
    T result[M];
    static constexpr ACCL::dataType datapath_type = (TYPE == 0) ? ACCL::dataType::int32 : ACCL::dataType::float32;
    std::unique_ptr<ACCL::Buffer<T>> acc_buf, res_buf;
    if (options.hardware && options.rdma) {
	    acc_buf = accl.create_coyotebuffer<T>(M, datapath_type);
        res_buf = accl.create_coyotebuffer<T>(M, datapath_type);
    } else {
        acc_buf = accl.create_buffer<T>(M, datapath_type);
        res_buf = accl.create_buffer<T>(M, datapath_type);
    }

    auto application_begin_time = std::chrono::high_resolution_clock::now();

    // Read weights and inputs for this rank; initialize outputs
    // TODO - See if it can be optimised further
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix(M, N/WS);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> data(N/WS, 1);
    for (int j = 0 ; j < M ; j++) {
        for (int k = 0 ; k < N/WS ; k++) {
            matrix(j, k) = weights[j][current_rank * (N/WS) + k]; 
            data(k) = in[current_rank * (N/WS) + k];
        }
    }

    accl.barrier();
    auto computation_begin_time = std::chrono::high_resolution_clock::now();

    // Partial MVM, with data/weights for this rank
    auto product = (matrix * data).eval();
    auto computation_end_time = std::chrono::high_resolution_clock::now();

    // Store into buffers suitable for ACCL REDUCE
    for (int i = 0; i < M; i++) {
        acc_buf.get()->buffer()[i] = product(i, 0);
    }

    // Barrier to ensure everyone set up & reduce
    ACCL::ACCLRequest* req;
    if (options.hardware && options.rdma) {
        req = accl.reduce(*acc_buf, *res_buf, M, root, ACCL::reduceFunction::SUM, ACCL::GLOBAL_COMM, true, true, ACCL::dataType::none, true);
        accl.wait(req, 1500ms);
    } else {
        req = accl.reduce(*acc_buf, *res_buf, M, root, ACCL::reduceFunction::SUM);
        accl.wait(req, 1500ms);
   }
    auto reduce_time = std::chrono::high_resolution_clock::now();

    // Store to "local" memory
    for (int i = 0 ; i < M ; i++) {
        result[i] = res_buf.get()->buffer()[i];
    }
    auto application_end_time = std::chrono::high_resolution_clock::now();

    // Write outputs to text file, to be compared with target output later
    if (current_rank == root) {
        std::ofstream f("../log/accl_log/accl_outputs.txt");
        if (f.is_open()) {
            for (int i = 0; i < M; i++) {
                // Avoid extra space at the end, complicates parsing in Python
                std::string delim = (i == M - 1) ? "" : " "; 
                f << result[i] << delim;
            }
            f.close();
        } 
    }

    // Output some timing information
    if (root == current_rank) {
        std::cout << "MVM: Computation duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(computation_end_time - computation_begin_time).count() << "ns" << std::endl;
        std::cout << "MVM: Results available on root duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(reduce_time - computation_begin_time).count() << "ns" << std::endl;
        std::cout << "MVM: Application duration " << std::chrono::duration_cast<std::chrono::nanoseconds>(application_end_time - application_begin_time).count() << "ns" << std::endl;
    }

    // Free buffers & finish program
    acc_buf->free_buffer();
    res_buf->free_buffer();
}

// Application code runs on the FPGA, as a kernel
void run_with_kernel(
    ACCL::ACCL &accl,
    int current_rank,
    options_t options
) {
    // Read weights for this rank, and, initialise empty results array
    T res[M];
    T W[M][N/WS];
    for (int j = 0 ; j < M ; j++) {
        for (int k = 0 ; k < N/WS ; k++) {
            W[j][k] = weights[j][current_rank * (N/WS) + k]; 
        }
    }

    // Hardware test
    if (options.hardware) {
        exit(1);    // TODO - Implement
    // Simulation/Emulation
    } else {
        // Initialize a CCLO BFM (emulation/simulation only) and streams as needed
        hlslib::Stream<command_word> callreq, callack;
        hlslib::Stream<stream_word, 512> data_cclo2krnl, data_krnl2cclo;
        std::vector<unsigned int> dest = {0}; 
        CCLO_BFM cclo(options.start_port, current_rank, WS, dest, callreq, callack, data_cclo2krnl, data_krnl2cclo);
        cclo.run();

        // Barrier to make sure all ranks have finished setting up
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Run kernel
        auto config_addr = (TYPE == 0) ? 
                            accl.get_arithmetic_config_addr({ACCL::dataType::int32, ACCL::dataType::int32}) :
                            accl.get_arithmetic_config_addr({ACCL::dataType::float32, ACCL::dataType::float32});
        mvm(   
            in + current_rank * N/WS, res, W, current_rank,
            accl.get_communicator_addr(), config_addr,
            callreq, callack, data_krnl2cclo, data_cclo2krnl
        );

        // Stop CCLO BFM
        cclo.stop();
    }

    // Write outputs to text file, to be compared with target output later
    if (current_rank == root) {
        std::ofstream f("../log/accl_log/accl_outputs.txt");
        if (f.is_open()) {
            for (int i = 0; i < M; i++) {
                // Avoid extra space at the end, complicates parsing in Python
                std::string delim = (i == M - 1) ? "" : " "; 
                f << res[i] << delim;
            };
            f.close();
        }
    }

}

/////////////////////////////////////////////
//              CONFIGURATION             //
///////////////////////////////////////////
void exchange_qp(
    unsigned int master_rank, 
    unsigned int slave_rank, 
    unsigned int local_rank, 
    std::vector<fpga::ibvQpConn*> &ibvQpConn_vec, 
    std::vector<ACCL::rank_t> &ranks
) {
	if (local_rank == master_rank) {
		std::cout << "Local rank " << local_rank << " sending local QP to remote rank " << slave_rank << std::endl;
		
        // Send the local queue pair information to the slave rank
		MPI_Send(&(ibvQpConn_vec[slave_rank]->getQpairStruct()->local), sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0, MPI_COMM_WORLD);
	}
	else if (local_rank == slave_rank) {
		std::cout << "Local rank " << local_rank << " receiving remote QP from remote rank " << master_rank << std::endl;
		
        // Receive the queue pair information from the master rank
		fpga::ibvQ received_q;
		MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
        // Copy the received data to the remote queue pair
		ibvQpConn_vec[master_rank]->getQpairStruct()->remote = received_q;
	}

	// Synchronize after the first exchange to avoid race conditions
	MPI_Barrier(MPI_COMM_WORLD);

	if (local_rank == slave_rank) {
		std::cout << "Local rank " << local_rank << " sending local QP to remote rank " << master_rank << std::endl;
		
        // Send the local queue pair information to the master rank
		MPI_Send(&(ibvQpConn_vec[master_rank]->getQpairStruct()->local), sizeof(fpga::ibvQ), MPI_CHAR, master_rank, 0, MPI_COMM_WORLD);
	}
	else if (local_rank == master_rank) {
		std::cout << "Local rank " << local_rank << " receiving remote QP from remote rank " << slave_rank << std::endl;
		
        // Receive the queue pair information from the slave rank
		fpga::ibvQ received_q;
		MPI_Recv(&received_q, sizeof(fpga::ibvQ), MPI_CHAR, slave_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		// Copy the received data to the remote queue pair
		ibvQpConn_vec[slave_rank]->getQpairStruct()->remote = received_q;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Write established connection to hardware and perform arp lookup
	if (local_rank == master_rank)
	{
		int connection = (ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn & 0xFFFF) | ((ibvQpConn_vec[slave_rank]->getQpairStruct()->remote.qpn & 0xFFFF) << 16);
		ibvQpConn_vec[slave_rank]->getQpairStruct()->print();
		ibvQpConn_vec[slave_rank]->setConnection(connection);
		ibvQpConn_vec[slave_rank]->writeContext(ranks[slave_rank].port);
		ibvQpConn_vec[slave_rank]->doArpLookup();
		ranks[slave_rank].session_id = ibvQpConn_vec[slave_rank]->getQpairStruct()->local.qpn;
	} else if (local_rank == slave_rank) {
		int connection = (ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn & 0xFFFF) | ((ibvQpConn_vec[master_rank]->getQpairStruct()->remote.qpn & 0xFFFF) << 16);
		ibvQpConn_vec[master_rank]->getQpairStruct()->print();
		ibvQpConn_vec[master_rank]->setConnection(connection);
		ibvQpConn_vec[master_rank]->writeContext(ranks[master_rank].port);
		ibvQpConn_vec[master_rank]->doArpLookup();
		ranks[master_rank].session_id = ibvQpConn_vec[master_rank]->getQpairStruct()->local.qpn;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

// Helper function to configure RDMA network
void configure_cyt_rdma(std::vector<ACCL::rank_t> &ranks, int current_rank, ACCL::CoyoteDevice* device) {
	std::cout << "Initializing QP connections..." << std::endl;
	
    // Create queue pair connections
	std::vector<fpga::ibvQpConn*> ibvQpConn_vec;
	
    // Create single page dummy memory space for each qp
	uint32_t n_pages = 1;
	for (int i = 0; i < ranks.size(); i++) {
		fpga::ibvQpConn* qpConn = new fpga::ibvQpConn(device->coyote_qProc_vec[i], ranks[current_rank].ip, n_pages);
		ibvQpConn_vec.push_back(qpConn);
	}

	std::cout << "Exchanging QP..." << std::endl;
	for (int i = 0; i < ranks.size(); i++) {

		for (int j = i + 1; j < ranks.size(); j++) {
			exchange_qp(i, j, current_rank, ibvQpConn_vec, ranks);
		}
	}
    std::cout << "Done..." << std::endl;

}

// Helper function to configure TCP network
void configure_cyt_tcp(std::vector<ACCL::rank_t> &ranks, int current_rank, ACCL::CoyoteDevice* device) {
	std::cout << "Configuring Coyote TCP..." << std::endl;
	
    // Arp lookup
    for (int i = 0; i < ranks.size(); i++){
        if (current_rank != i){
            device->get_device()->doArpLookup(_ip_encode(ranks[i].ip));
        }
    }

	// Open port 
    for (int i = 0; i<ranks.size(); i++) {
        uint32_t dstPort = ranks[i].port;
        bool open_port_status = device->get_device()->tcpOpenPort(dstPort);
    }

	std::this_thread::sleep_for(10ms);

	// Open connection
    for (int i = 0; i < ranks.size(); i++) {
        uint32_t dstPort = ranks[i].port;
        uint32_t dstIp = _ip_encode(ranks[i].ip);
        uint32_t dstRank = i;
		uint32_t session = 0;
        if (current_rank != dstRank) {
            bool success = device->get_device()->tcpOpenCon(dstIp, dstPort, &session);
			ranks[i].session_id = session;
        }
    }

}

// Helper function, initialising ACCL, depending if it runs on the hardware or not
std::unique_ptr<ACCL::ACCL> initialise_accl(options_t options, int current_rank) {
    // Simulation 
    if (!options.hardware) {
        // Generate a list of ranks in the world
        std::vector<ACCL::rank_t> ranks;
        ranks = accl_network_utils::generate_ranks(true, current_rank, WS, options.start_port, options.rxbuf_size);

        // Set communication protocol
        accl_network_utils::acclDesign design;
        if (options.udp) {
            design = accl_network_utils::acclDesign::UDP;
        } else if (options.tcp) {
            design = accl_network_utils::acclDesign::TCP;
        } else if (options.rdma) {
            design = accl_network_utils::acclDesign::ROCE;
        }

        // Initialize ACCL
        std::unique_ptr<ACCL::ACCL> accl = accl_network_utils::initialize_accl(
            ranks, current_rank, !options.hardware, design
        );
        accl->set_timeout(1e6);
        return accl;
    
    // Hardware test
    } else {
        // Generate a list of ranks in the world
        std::vector<ACCL::rank_t> ranks;
	
    	// Load IP addresses for ranks
	    std::ifstream file;
        file.open(options.fpgaIP);
        if (!file.is_open()) {
            perror("Error opening fpgaIP file");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < WS; i++) {
		    std::string ip;
			getline(file, ip);
			std::cout << ip << std::endl;

            // Generate rank
            if(options.rdma) {
                ACCL::rank_t new_rank = {ip, options.start_port, i, options.rxbuf_size};
                ranks.emplace_back(new_rank);
            } else {
                ACCL::rank_t new_rank = {ip, options.start_port + i, 0, options.rxbuf_size};
                ranks.emplace_back(new_rank);
            }      
        }

        // Initialise Coyote device
        ACCL::CoyoteDevice* device;
        if (options.tcp){
            device = new ACCL::CoyoteDevice();
        } else if (options.rdma){
            device = new ACCL::CoyoteDevice(WS);
        }
	
        // Configure network protocol
        if (options.tcp){
		    configure_cyt_tcp(ranks, current_rank, device);
	    } else if (options.rdma){
            configure_cyt_rdma(ranks, current_rank, device);
        } else {
            std::cout << "ERROR: UDP NOT SUPPORTED" << std::endl;
			exit(1);
        }
    	
        // Rendezvouz protocol
        std::unique_ptr<ACCL::ACCL> accl = std::make_unique<ACCL::ACCL>(
            device, ranks, current_rank,
            WS, 32, 64, 4096 * 1024
        );

		ACCL::debug(accl->dump_communicator());
        return accl;
    }
}

// No operation test
void test_nop(ACCL::ACCL &accl, int current_rank) {
    // Measure time
	auto start = std::chrono::high_resolution_clock::now();	
    ACCL::ACCLRequest* req = accl.nop(true);
  	accl.wait(req);
	auto end = std::chrono::high_resolution_clock::now();
	
    // Print
    double durationUs = (std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count() / 1000.0);
	uint64_t durationNs = accl.get_duration(req);
	std::cout << "SW NoP time [us]:"<< durationUs << std::endl;
	std::cout << "HW NoP time [ns]:"<< std::dec << durationNs << std::endl;

    // Debug log
	std::cerr << "Rank " << current_rank << " passed last barrier before test!" << std::endl << std::flush;
}

/////////////////////////////////////////////
//               MAIN                     //
///////////////////////////////////////////
int main(int argc, char *argv[]) {    
    // Parse CLI arguments
    options_t options = parse_options(argc, argv);
    
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

    // Initialise ACCL
    std::unique_ptr<ACCL::ACCL> accl = initialise_accl(options, current_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Final test before starting benchmark
    if (options.hardware) {
        test_nop(*accl, current_rank);
        accl->barrier();
    }
    
    // Run test
    if (options.kernel) {
        run_with_kernel(*accl, current_rank, options);
    } else {
        run_on_host_cluster(*accl, current_rank, options);
    }

    // Delete ACCL unique_ptr
    MPI_Barrier(MPI_COMM_WORLD);
    accl.reset();

    // Cleans up the MPI environment and ends MPI communications
    MPI_Finalize();
}
