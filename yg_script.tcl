############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2020 Xilinx, Inc. All Rights Reserved.
############################################################
open_project hls_prj
set_top sa_compute
add_files block_top.cpp
add_files block_top.h
add_files config_sa.h
add_files matrix-vector-unit-dsp.h
add_files misc.h
add_files misc_DSP.h
add_files param.h
add_files pipeSA_tools.h
add_files rocky-lib.h
add_files stream.h
add_files -tb block_test.cpp
add_files -tb conv2_hls_out_4bit.txt
open_solution "s1" -flow_target vivado
set_part {xczu9eg-ffvb1156-2-e}
create_clock -period 3.3 -name default
config_export -format ip_catalog -rtl verilog
#source "./hls_prj/s1/directives.tcl"
#csim_design -clean -setup
#csynth_design
#cosim_design -trace_level port
#export_design -flow impl -rtl verilog -format ip_catalog