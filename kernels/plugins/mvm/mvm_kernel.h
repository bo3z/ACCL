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

#ifndef MVM_KERNEL_H
#define MVM_KERNEL_H

#include <accl_hls.h>
#include <mvm_constants.h>

/*
    Generic application kernel for column-wise matrix-vector multiply (MVM)
    Matrix dimensionality MxN, input vector dimensionality Nx1
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
);

#endif
