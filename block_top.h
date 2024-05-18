#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <hls_stream.h>
#include "stream.h"
#include "config_sa.h"
using namespace hls;

void sa_compute(stream<trans_pkt_in >& in,stream<trans_pkt_out >& out);