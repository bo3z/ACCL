#############################################################################
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
#############################################################################

# TODO - Floats are still buggy

import re
import os
import json
import argparse
import numpy as np

###############################
##  RANDOM GENERATOR HELPERS ##
###############################
def __generate_random_tensor(L, dtype, size):
    '''
    Generates a random tensor of length N, with maximum vaue L and type=dtype
    '''
    if dtype == 'float':
        return 2 * L * np.random.random(size) - L
    elif dtype == 'int':
        return np.random.randint(-L, L, size=size)
    else:
        raise Exception('Unknown data type...')

def generate_random_vector(N, L, dtype):
    '''
    Generates a random vector of length N, with maximum vaue L and type=dtype
    '''
    return __generate_random_tensor(L, dtype, size=(N, ))

def generate_random_matrix(M, N, L, dtype):
    '''
    Generates a random matrix, dimension MxN, with maximum vaue L and type=dtype
    '''
    return __generate_random_tensor(L, dtype, size=(M, N))

###############################
##  C++ SERIALIZING HELPERS  ##
###############################
def save_npvec_tocpp_array(vec, dtype, name='vec'):
    '''
    Helper function that parses a NumPy array and returns a string description of the same array in C++
    
    Args:
    vec (np.array) - Array to be serialised
    dtype (string) - int or float
    name (string) - Name of the variable in C++
    '''
    return f'static constexpr {dtype} {name}[{vec.shape[0]}] = {{{str(vec.tolist())[1:-1]}}};\n'

def save_npmat_tocpp_array(mat, dtype, name='mat'):
    '''
    Helper function that parses a matix array and returns a string description of the same array in C++
    
    Args:
    mat (np.array) - Matrix to be serialised
    dtype (string) - int or float
    name (string) - Name of the variable in C++
    '''
    line = f'static constexpr {dtype} {name}[{mat.shape[0]}][{mat.shape[1]}] = {{\n'
    for v in mat:
        line += f'\t{{{str(v.tolist())[1:-1]}}},\n'
    line += f'}};\n'
    return line;

###############################
##  TIMING BENCHMARK HELPERS ##
###############################
def store_data_to_json(root_rank, n, M, N, test_mode):
    '''
    Helper function that parses the outputs from log/* and stores the time taken
    
    Args:
    root_rank (int) - Destination rank
    n (int) - World size
    M (int) - Number of outputs
    N (int) - Number of inputs
    test_mode (str) - accl or openmpi

    Note: World size acts as a unique identifier for the test (want to compare latency wih varying world size)
    '''
    TARGET_UNIT = 'ms'                  # Target time unit 
    FACTOR = 1000 * 1000                # Divider from ns to target unit
    JSON_OUT_FILE = 'results.json' # File to store the results, for future plotting

    # Empty list, populate with values for every trial
    comp_done_time = []
    res_available_time = []
    full_pipeline_time = []

    # Regex patterns
    pattern_comp = r'MVM: Computation duration ([0-9]+)ns'
    pattern_full = r'MVM: Application duration ([0-9]+)ns'
    pattern_res = r'MVM: Results available on root duration ([0-9]+)ns'

    # Identify directories for each test
    trial_dirs = next(os.walk(f'../log/{test_mode}_log'))[1]
    for dir in trial_dirs:
        # Since we use different MPI versions for ACCL and OpenMPI, the output file names will differ
        if test_mode == 'accl':
            file_name = f'../log/{test_mode}_log/{dir}/rank_{root_rank}_stdout.log'
        else:
            file_name = f'../log/{test_mode}_log/{dir}/1/rank.{root_rank}/stdout'
        
        # Go through all the lines and find the ones matching the time measurement
        for _, line in enumerate(open(file_name)):
            match_comp = re.match(pattern_comp, line)
            if match_comp:
                comp_done_time.append(int(match_comp.group(1)))

            match_res = re.match(pattern_res, line)
            if match_res:
                res_available_time.append(int(match_res.group(1)))

            match_full = re.match(pattern_full, line)
            if match_full:
                full_pipeline_time.append(int(match_full.group(1)))
    
    # Calculate mean time and onvert to milliseconds
    results = {
        f'world_size_{n}_M_{M}_N_{N}_mode_{test_mode}': {
            'computation_mean': np.mean(comp_done_time) / FACTOR,
            'result_available_mean': np.mean(res_available_time) / FACTOR,
            'pipeline_complete_mean': np.mean(full_pipeline_time) / FACTOR,
            'root_rank': root_rank,
            'world_size': n,
            'n_tests': len(trial_dirs),
            'time_unit': TARGET_UNIT
        }
    }

    # Include any previous results
    if (os.path.exists(JSON_OUT_FILE)):
        with open(JSON_OUT_FILE, 'r') as j:
            results.update(json.loads(j.read()))

    # Store to JSON file for future plotting
    with open(JSON_OUT_FILE, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    ###################################
    #               CLI              #
    ##################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, help='Number of ranks', required=True)
    parser.add_argument('-N', '--inputs', type=int, help='Number of elements in input vector', required=True)
    parser.add_argument('-M', '--outputs', type=int, help='Number of elements in output vector', required=True)
    parser.add_argument('-L', '--largest', type=int, help='Largest element vector/matrix', required=True)
    parser.add_argument('-k', '--kernel', help='Run application as FPGA kernel', action='store_true')
    parser.add_argument('-f', '--hardware', help='Test on physical hardware', action='store_true')
    parser.add_argument('-a', '--accl', help='Run ACCL test', action='store_true')
    parser.add_argument('-o', '--openmpi', help='Run OpenMPI test', action='store_true')
    parser.add_argument('-r', '--root', type=int, help='Root (destination) rank', default=0)
    parser.add_argument('-T', '--dtype', type=str, help='Data type: int or float', choices={'int', 'float'}, default='int')
    parser.add_argument('-c', '--comms', type=str, help='Communication protocol', choices={'tcp', 'udp', 'rdma'}, default='rdma')
    parser.add_argument('-t', '--tests', type=int, help='Number of repeated tests', default=3)

    args = parser.parse_args()    
    n = args.number
    N = args.inputs
    M = args.outputs
    L = args.largest
    comms = args.comms
    kernel = args.kernel
    hardware = args.hardware
    accl = args.accl
    openmpi = args.openmpi
    root = args.root
    dtype = args.dtype
    tests = args.tests

    # Check if hardware => RDMA
    # Note, TCP/IP & UDP can be tested on physical FPGAs, but it requires rebuilding the project from scratch
    # For more info, see ACCL/test/refdesigns/Makefile
    if hardware and comms != 'rdma':
        raise Exception('Hardware mode enabled without RDMA!')

    if N % n != 0:
        raise Exception('Number of ranks must divide number of inputs for correct functionality')
    
    # Folder housekeeping
    os.system('rm -r -f ../log && mkdir ../log && mkdir ../log/accl_log && mkdir ../log/openmpi_log')

    ###################################
    #           Server IDs           #
    ##################################
    if hardware:
        print('Enter Alveo u55c IDs:')
        ids = input()
        if len(ids) != 2 * n - 1:
            raise Exception('Enter exact number of IDs')

    # Repeat test several times
    for i in range(1, tests + 1):
        ###################################
        #         TESTING DATA            #
        ##################################
        # Generate random vector/matrix for testing
        data = generate_random_vector(N, L, dtype)
        weights = generate_random_matrix(M, N, L, dtype)
        
        # Save data to C++ header file
        with open('../data/mvm_data.h', 'w') as f:
            f.write('#ifndef MVM_DATA_H\n')
            f.write('#define MVM_DATA_H\n')
            f.write('\n')
            f.write(save_npvec_tocpp_array(data, dtype, 'in'))
            f.write(save_npmat_tocpp_array(weights, dtype, 'weights'))
            f.write('#endif\n')
    
        # Save config to C++ constants file
        with open('../data/mvm_constants.h', 'w') as f:
            f.write('#ifndef MVM_CONSTANTS_H\n')
            f.write('#define MVM_CONSTANTS_H\n')
            f.write('\n')
            f.write(f'typedef {dtype} T; // Data type\n')
            f.write(f'static constexpr int WS = {n}; // World size\n')
            f.write(f'static constexpr int M = {M}; // Number of outputs\n')
            f.write(f'static constexpr int N = {N}; // Number of inputs\n')
            f.write(f'static constexpr int root = {root}; // Root rank\n')
            # Additional variable for datapath configuration
            type_str = 0 if dtype == 'int' else 1
            f.write(f'static constexpr int TYPE = {type_str}; \n')
            f.write('#endif\n')
    
        ###################################
        #           ACCL TEST            #
        ##################################
        if accl:
            make = os.system(f'cd accl && /bin/cmake . && make')
            if make > 0:
                raise Exception('ACCL compilation failed')
            
            # Make sure emulator/simulator is started beforehand
            if not hardware:
                flags = ''

                if comms == 'tcp':
                    flags += '-t '
                elif comms == 'rdma':
                    flags += '-r '
                elif comms == 'udp':
                    flags += '-u '
                
                if kernel:
                    flags += '-k '
                
                if hardware:
                    flags += '-f'

                print(f'Starting simulation with {n} rank(s)')
                os.system(f'cd accl && mpirun -np {n} bin/test_accl {flags}')
            
            # Make sure FPGA is programmed beforehand (scripts/flow_u55c.sh)
            else:
                # Deteremine the RX Buffer Size (in KB), needed to run correctly
                # floats / ints are 4 bytes, so to reduce the full results need M * 4 bytes
                # Conver to KB, by dividing with 1024
                # However, max RX Buffer Size is 4 MB
                # Also, the buffer size cannot be 0, so use 1KB
                # If needed, a higher level of granularity can be added, e.g. 512B
                buf_size = int(min(max(1, M * 4 / 1024), 4096))

                # Run test
                print(f'Starting hardware test with {n} rank(s) using ACCL')
                print(f'cd ../scripts && bash run_accl.sh {buf_size} {str(i)} {ids}')
                os.system(f'cd ../scripts && bash run_accl.sh {buf_size} {str(i)} {ids}')

            # Load ACCL results and compare to NumPy
            accl_out = np.loadtxt('../log/accl_log/accl_outputs.txt', delimiter=' ', dtype=dtype) 
            np.testing.assert_array_equal(accl_out, weights @ data)
            print(f'TEST {i} PASSED WITH NO ERRORS USING ACCL!')

            # Housekeeping
            os.system('rm ../log/accl_log/accl_outputs.txt')

        ###################################
        #          OPENMPI TEST           #
        ##################################
        # Make sure emulator/simulator is started beforehand
        if openmpi:
            make = os.system(f'cd openmpi && /bin/cmake . && make')
            if make > 0:
                raise Exception('OpenMPI compilation failed')

            print(f'Starting hardware test with {n} rank(s) using OpenMPI')
            os.system(f'cd ../scripts && bash run_openmpi.sh {str(i)} {ids}')

            # Load ACCL results and compare to NumPy
            openmpi_out = np.loadtxt('../log/openmpi_log/openmpi_outputs.txt', delimiter=' ', dtype=dtype) 
            np.testing.assert_array_equal(openmpi_out, weights @ data)
            print(f'TEST {i} PASSED WITH NO ERRORS USING OPENMPI!')

            # Housekeeping
            os.system('rm ../log/openmpi_log/openmpi_outputs.txt')

    # Store data for future plotting
    if hardware and accl:
        store_data_to_json(root, n, M, N, 'accl')
    if hardware and openmpi:
        store_data_to_json(root, n, M, N, 'openmpi')
