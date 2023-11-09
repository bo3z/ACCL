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

DEVICE=xcu55c-fsvh2892-2L-e
MVM_IP=mvm.xo
TARGET=ip

all: $(MVM_IP) 

$(MVM_IP): build_mvm.tcl mvm_kernel.cpp
	vitis_hls $< -tclargs $(TARGET) $(DEVICE)

