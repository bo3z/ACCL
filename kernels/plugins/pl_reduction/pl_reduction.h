/*******************************************************************************
#  Copyright (C) 2022 Advanced Micro Devices, Inc
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

#include <accl_hls.h>

// N - Array length in each rank
// In each rank, the kernal generates an array, i-th element is equal to i * rank
// Then, the expected sum at the root rank, is such that the i-th element is i * (0 + ... + ranks - 1) = i * ranks * (ranks - 1) / 2
template<int N>
void pl_reduce(
    // Rank of the current kernel - TODO: add HLS connection
    int rank, 

    // Total number of ranks
    int ranks,

    // Parameters pertaining to CCLO config
    ap_uint<32> reduce_addr, 
    ap_uint<32> dpcfg_adr,
    
    // Streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
) {
    #pragma HLS INTERFACE s_axilite port=rank
    #pragma HLS INTERFACE s_axilite port=ranks

    #pragma HLS INTERFACE m_axi port=in offset=slave
    #pragma HLS INTERFACE m_axi port=weights offset=slave
    
    #pragma HLS INTERFACE s_axilite port=gather_adr
    #pragma HLS INTERFACE s_axilite port=dpcfg_adr
    
    #pragma HLS INTERFACE axis port=cmd_to_cclo
    #pragma HLS INTERFACE axis port=sts_from_cclo
    #pragma HLS INTERFACE axis port=data_to_cclo
    #pragma HLS INTERFACE axis port=data_from_cclo

    // Some constants
    constexpr int strm_flg = 3; 
    constexpr int dst_rank = 0;

    // Initialize CCLO
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo, reduce_addr, dpcfg_adr, 0, strm_flg);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    
    // Generate array
    int in[N] = {0};
    for (int i = 0 ; i < N ; i++) {
        #pragma HLS unroll
        in[i] = i * rank;
    }

    // Read data from array and push the result to the CCLO stream
    ap_int<512> word;
    for(int i = 0; i < N; i++) {
        #pragma HLS unroll
        // Correct implementation
        // word((i + 1) * 32 - 1, i * 32) = *reinterpret_cast<ap_uint<32>*>(&in[i]);

        // Incorrect
        word(i * 32, (i + 1) * 32 - 1) = *reinterpret_cast<ap_uint<32>*>(&in[i]);

    }
    data.push(word, 0);

    // Reduce stream-to-stream with SUM
    accl.reduce(N, dst_rank, 0);
    
    // Read reduced data and check true
    if (rank == dst_rank) {
        word = data.pull().data;
        for(int i = 0; i < N; i++) {
            #pragma HLS unroll
            // Correct implementation
            // ap_int<32> val = word((i + 1) * 32 - 1, i * 32);

            // Incorrect
            ap_int<32> val = word(i * 32, (i + 1) * 32 - 1);
            
            int x = *reinterpret_cast<int*>(&val);
            bool equal = x == i * ranks * (ranks - 1) / 2;
            std::cout << "Equal: " << equal << ", reduced=" << x << ", expected=" << i * ranks * (ranks - 1) / 2 << std::endl;
        }
    }
}