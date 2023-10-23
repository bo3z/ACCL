import os

# Remove from cclo
os.system(f'rm -r -f cclo/.Xil cclo/ccl_offload_ex cclo/vitis_ws cclo/*.jou cclo/*.log')

# Remove from plugins
plugins = [f.path for f in os.scandir('plugins/') if f.is_dir()]
for d in plugins:
    os.system(f'rm -r -f {d}/build_*/* {d}/*.xo {d}/*.log')
