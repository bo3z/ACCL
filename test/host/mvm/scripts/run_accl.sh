#!/bin/bash

# Optional variables
RX_BUF_SIZE=$1
shift
TRIAL_ID=$1
shift
SERVER_ID=("$@")

# Check working directory
if [[ $(pwd) != */test/host/mvm/scripts ]]; then
	echo "ERROR: This script needs to be run from /test/host/mvm/scripts"
	exit 1
fi

# State variables and output files
mkdir ../log/accl_log
rm -r -f ../log/accl_log/trial_$TRIAL_ID
mkdir ../log/accl_log/trial_$TRIAL_ID
EXEC=../source/accl/bin/test_accl
HOST_FILE=../log/host.txt
FPGA_FILE=../log/fpga.txt

# Create IP config files
rm -f $HOST_FILE $FPGA_FILE
NUM_PROCESS=0
for ID in ${SERVER_ID[@]}; do
	echo "10.253.74.$(((ID - 1) * 4 + 66))" >> $HOST_FILE
	echo "10.253.74.$(((ID - 1) * 4 + 68))" >> $FPGA_FILE
	NUM_PROCESS=$((NUM_PROCESS + 1))
	HOST_LIST+="alveo-u55c-$(printf "%02d" $ID) "
done

# Run
ARG="-f -r" # Hardware and TCP/RDMA flags
echo "mpirun -n $NUM_PROCESS -f $HOST_FILE --iface ens4 $EXEC $ARG -l $FPGA_FILE"
mpirun -n $NUM_PROCESS -f $HOST_FILE --iface ens4f0 -outfile-pattern "../log/accl_log/trial_$TRIAL_ID/rank_%r_stdout.log" -errfile-pattern "../log/accl_log/trial_$TRIAL_ID/rank_%r_stdout.log" $EXEC $ARG -l $FPGA_FILE -b $RX_BUF_SIZE

# Wait to complete
SLEEPTIME=5
sleep $SLEEPTIME

# Terminate the process
parallel-ssh -H "$HOST_LIST" "killall -9 -u bramhorst test_accl"
exit 0
