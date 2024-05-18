#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const

#include "block_top.h"
#include <ap_axi_sdata.h>
#include <cstdlib>
#include <stdio.h>
#include <fstream>
#include <iostream>


#define BIT         4      //输入比特数
// #define MAX_INP 32
#define SIZE 56*56*64


bool loadFile_txt(const char* name1, int reps,stream<trans_pkt_in >& in);


int main(void){



	trans_pkt_in in_tmp;
	int reps=1;


	stream<trans_pkt_in > in_stream("in_stream");
	loadFile_txt("conv2_hls_out_4bit.txt", 1,in_stream);

	stream<trans_pkt_out > out_stream("out_stream");

	sa_compute(in_stream, out_stream);

	trans_pkt_out temp_stream;
	 FILE* fp1 = fopen("result_verify_conv8.txt", "wb");

	 	ap_uint<16> tmp_out;
	 	ap_uint<256> out_data;

	 	for (unsigned i = 0; i < SIZE/16; i++) {
	 		temp_stream=out_stream.read();
			out_data=temp_stream.data(256-1,0);
	 		for(int j=0;j<16;j++){      //out channel
	 			tmp_out=out_data(16-1,0);
	 			fprintf(fp1, "%lf\n", double(tmp_out));
	 			out_data=out_data>>16;
	 		}
	 	}
	 	fclose(fp1);

	return 0;
}



bool loadFile_txt(const char* name1, int reps,stream<trans_pkt_in >& in){
	FILE* fp1 = fopen(name1, "rb");
	int i = 0;
	int j = 0;
	ap_uint<4> tmp;  //输入为3bit
	ap_uint<256> in_data;
	trans_pkt_in in_tmp;
	int rep;
	double temp;
	FILE* fp_input = fopen("input_verfy.txt", "wb");
	if ((fp1 == NULL)) {
		std::cout << "Load Error!" << std::endl;
		return false;
	}
	for(rep=0;rep<reps;rep++){
		for (i = 0; i<SIZE/(32*2); i++) {
			for (j = 0; j<MAX_INP*2; j++) {
				fscanf(fp1, "%lf", &temp);  //数据格式为double
				
				tmp = (ap_uint<4>)temp;  //数据转化为输入bit
				fprintf(fp_input, "%lf\n", double(tmp));
	//			cout << "start:" << tmp <<endl;
				in_data(BIT*(j+1)-1, BIT*j) = tmp;
			}
			in_tmp.data = in_data;
			in.write(in_tmp);
		}
	}
	fflush(fp1);  //清除读写缓冲区。强迫将缓冲区的数据写回参数stream指定的文件中
	fclose(fp1);
	fclose(fp_input);
	std::cout << "Load Success!" << std::endl;
	return true;
}










