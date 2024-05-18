#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <ap_int.h>

using namespace hls;
using namespace std;





#define DWIDTH_IN 256
#define DWIDTH_OUP 256
typedef ap_axiu<DWIDTH_IN,0,0,0> trans_pkt_in;
typedef ap_axiu<DWIDTH_OUP,0,0,0> trans_pkt_out;






