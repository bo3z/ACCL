#!/bin/bash

# Optional variables
TRIAL_ID=$1
shift
SERVER_ID=("$@")

# Check bash version
if [  "${BASH_VERSINFO:-0}" -lt 4 ]; then
  echo "Script requires bash v4 or newer; you're using bash v${BASH_VERSION}"
  echo "Exiting..."
  exit 1
fi

# Check working directory
if [[ $(pwd) != */test/host/mvm/scripts ]]; then
	echo "ERROR: This script needs to be run from /test/host/mvm/scripts"
	exit 1
fi

mellanox_name="mlx5_0"
echo "Using mellanox card $mellanox_name"

# State variables and output files
rm -r -f ../log/openmpi_log/trial_$TRIAL_ID
mkdir ../log/openmpi_log/trial_$TRIAL_ID
EXEC=../source/openmpi/bin/test_openmpi
HOST_FILE=../log/host.txt

# Create IP config files
NUM_PROCESS=0
rm -f $HOST_FILE
for ID in ${SERVER_ID[@]}; do
  echo "10.253.74.$(((ID - 1) * 4 + 66))" >> $HOST_FILE
  NUM_PROCESS=$((NUM_PROCESS+1))
done

mpirun --prefix /mnt/scratch/trlaan/openmpi -output-filename "../log/openmpi_log/trial_$TRIAL_ID" -np $NUM_PROCESS \
        --mca btl '^openib' --mca pml ucx -x UCX_NET_DEVICES=$mellanox_name:1 -x PATH -x LD_LIBRARY_PATH --hostfile $HOST_FILE $EXEC
