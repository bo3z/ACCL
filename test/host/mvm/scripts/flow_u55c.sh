#!/bin/bash

# Parameters
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FPGA_BIT_PATH=/tmp/bitstreams/cyt_top
DRIVER_PATH=/mnt/scratch/bramhorst/driver

# Helper to program Alveo 
alveo_program()
{
	SERVERADDR=$1
	SERVERPORT=$2
	BOARDSN=$3
	DEVICENAME=$4
	BITPATH=$5
	vivado -nolog -nojournal -mode batch -source $SCRIPT_DIR/program_alveo.tcl -tclargs $SERVERADDR $SERVERPORT $BOARDSN $DEVICENAME $BITPATH
}

# Check Vivado is available
if ! [ -x "$(command -v vivado)" ]; then
	echo "Vivado does not exist on the system." 
	exit 1
fi

# Server IDs (u55c)
# Read server ids from user
echo "Enter u55c machine ids (space separated):"
read -a SERVID

# Generate host name list
for servid in ${SERVID[@]}; do 
	hostlist+="alveo-u55c-$(printf "%02d" $servid) "
done

# Some constants...
# FPGA serial number
BOARDSN=(XFL1QOQ1ATTYA XFL1O5FZSJEIA XFL1QGKZZ0HVA XFL11JYUKD4IA XFL1EN2C02C0A XFL1NMVTYXR4A XFL1WI3AMW4IA XFL1ELZXN2EGA XFL1W5OWZCXXA XFL1H2WA3T53A)
# FPGA IP address
IPADDR=(0AFD4A44 0AFD4A48 0AFD4A4C 0AFD4A50 0AFD4A54 0AFD4A58 0AFD4A5C 0AFD4A60 0AFD4A64 0AFD4A68)
# FPGA MAC address
MACADDR=(000A350B22D8 000A350B22E8 000A350B2340 000A350B24D8 000A350B23B8 000A350B2448 000A350B2520 000A350B2608 000A350B2498 000A350B2528)

# STEP 1: Program FPGA
# Activate servers (login with password to enable the nfs home mounting)
echo "Activating server..."
parallel-ssh -H "$hostlist" -A -O PreferredAuthentications=password "echo Login success!"
# Enable hardware server
echo "Enabling Vivado hw_server..."
# This step will be timeout after 2 secs to avoid the shell blocking
# Some of the nodes have different versions of Vivado so try all of them until one works
parallel-ssh -H "$hostlist" -t 2 "source /tools/Xilinx/Vivado/2022.1/settings64.sh && hw_server &"
parallel-ssh -H "$hostlist" -t 2 "source /tools/Xilinx/Vivado/2022.2/settings64.sh && hw_server &"
parallel-ssh -H "$hostlist" -t 2 "source /tools/Xilinx/Vivado/2023.1/settings64.sh && hw_server &"

# Program FPGA
echo "Programming FPGA...$FPGA_BIT_PATH"
for servid in "${SERVID[@]}"; do
	boardidx=$(expr $servid - 1)
	alveo_program alveo-u55c-$(printf "%02d" $servid) 3121 ${BOARDSN[boardidx]} xcu280_u55c_0 $FPGA_BIT_PATH &
done
wait
echo "FPGA programmed...$FPGA_BIT_PATH"

# STEP2: Hot-reset Host 
# Put -x '-tt' (pseudo terminal) here f or sudo command
echo "Removing the driver..."
parallel-ssh -H "$hostlist" -x '-tt' "sudo rmmod coyote_drv"
echo "Hot resetting PCIe..."	
parallel-ssh -H "$hostlist" -x '-tt' 'upstream_port=$(/opt/sgrt/cli/get/get_fpga_device_param 1 upstream_port) && root_port=$(/opt/sgrt/cli/get/get_fpga_device_param 1 root_port) && LinkCtl=$(/opt/sgrt/cli/get/get_fpga_device_param 1 LinkCtl) && sudo /opt/sgrt/cli/program/pci_hot_plug 1 $upstream_port $root_port $LinkCtl'
echo "Hot-reset done."
echo "Loading driver..."
for servid in "${SERVID[@]}"; do
	boardidx=$(expr $servid - 1)
	host="alveo-u55c-$(printf "%02d" $servid)"
	ssh -q -tt $host "sudo insmod $DRIVER_PATH/coyote_drv.ko ip_addr_q0=${IPADDR[boardidx]} mac_addr_q0=${MACADDR[boardidx]}" &
done
wait
echo "Driver loaded."
echo "Getting permissions for fpga..."
parallel-ssh -H "$hostlist" -x '-tt' "sudo /opt/sgrt/cli/program/fpga_chmod 0"
echo "Done."

exit 0
