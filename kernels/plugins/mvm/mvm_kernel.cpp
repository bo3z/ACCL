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

#include <mvm_kernel.h>

/*
    Generic application kernel for column-wise matrix-vector multiply (MVM)
*/
void mvm(
    // Vector input, results and weight matrix
    const T in[N/WS],
    T res[M],
    const T weights[M][N/WS],

    // World information
    int current_rank,

    // Parameters pertaining to CCLO config
    ap_uint<32> global_addr, 
    ap_uint<32> dpcfg_adr,
    
    // Streams to and from CCLO
    STREAM<command_word> &cmd_to_cclo,
    STREAM<command_word> &sts_from_cclo,
    STREAM<stream_word> &data_to_cclo,
    STREAM<stream_word> &data_from_cclo
) {
    #pragma HLS INTERFACE m_axi port=in offset=slave
    #pragma HLS INTERFACE m_axi port=res offset=slave
    #pragma HLS INTERFACE m_axi port=weights offset=slave

    #pragma HLS INTERFACE s_axilite port=current_rank
    
    #pragma HLS INTERFACE s_axilite port=global_addr
    #pragma HLS INTERFACE s_axilite port=dpcfg_adr
    
    #pragma HLS INTERFACE axis port=cmd_to_cclo
    #pragma HLS INTERFACE axis port=sts_from_cclo
    #pragma HLS INTERFACE axis port=data_to_cclo
    #pragma HLS INTERFACE axis port=data_from_cclo

    // Size of the data types, in bits
    static constexpr int DATA_SIZE = sizeof(T) * 8;
    static constexpr int BUFFER_SIZE = 512 / DATA_SIZE;
    
    // Set up  ACCL interfaces
    accl_hls::ACCLCommand accl(cmd_to_cclo, sts_from_cclo);
    accl_hls::ACCLData data(data_to_cclo, data_from_cclo);
    
    // Initialization and temporary results
    T acc[M] = {0};
    #pragma HLS array_partition variable=acc type=complete 

    // Partially unfolded MVM
    for (int i = 0; i < M; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < N/WS; j++) {
            #pragma HLS unroll
            acc[i] += weights[i][j] * in[j];
        }
    }

    // Some auxilary variables
    ap_uint<512> word;
    int word_count = 0;
    int rd_wr_count = M;

    // Send data to CCLO
    while(rd_wr_count > 0){
        // Read array elements into a 512b word
        for(int i = 0; (i < BUFFER_SIZE) && (rd_wr_count > 0); i++){
            word((i + 1) * DATA_SIZE - 1, i * DATA_SIZE) = *reinterpret_cast<ap_uint<DATA_SIZE>*>(&acc[i + BUFFER_SIZE * word_count]);
            rd_wr_count--;
        }

        // Push data to CCLO buffer
        data.push(word, 0);
        word_count++;
    }
    
    // Reduce
    accl.start_call(
        ACCL_REDUCE, M, global_addr, root, 
        0, 0, dpcfg_adr, 0, 3, 0, 0, 0
    );
    accl.finalize_call();
    
    // Write outputs
    if (current_rank == root) {
        int rd_wr_count = M;
        word_count = 0;

        while(rd_wr_count > 0) {
            // Read vector from CCLO
            word = data.pull().data;

            // Read from the 512b word into individual array elements
            for(int i = 0; (i < BUFFER_SIZE) && (rd_wr_count > 0); i++){
                ap_uint<DATA_SIZE> val = word((i + 1) * DATA_SIZE - 1, i * DATA_SIZE);
                res[i + BUFFER_SIZE * word_count] = *reinterpret_cast<T*>(&val);
                rd_wr_count--;
            }
            word_count++;
        }
    }
}
