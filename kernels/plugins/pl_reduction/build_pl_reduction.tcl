# /*******************************************************************************
#  Copyright (C) 2021 Advanced Micro Devices, Inc
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

set command [lindex $argv 0]
set device [lindex $argv 1]

set do_syn 0
set do_export 0

switch $command {
    "syn" {
        set do_syn 1
    }
    "ip" {
        set do_syn 1
        set do_export 1
    }
    "all" {
        set do_syn 1
        set do_export 1
    }
    default {
        puts "Unrecognized command"
        exit
    }
}


open_project build_pl_reduction

add_files pl_reduction.h -cflags "-std=c++14 -I../../../driver/hls/ -I. -DACCL_SYNTHESIS"

set_top pl_reduction

open_solution sol1
config_export -format xo -library ACCL -output [pwd]/pl_reduction.xo

if {$do_syn} {
    set_part $device
    create_clock -period 4 -name default
    csynth_design
}

if {$do_export} {
    export_design
}

exit
