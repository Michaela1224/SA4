#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
/*
 * 2021/03/10 已经查证
 */
template <	unsigned TopLeftPad,
			unsigned BottomRightPad,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void SAMEPAD(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps)
{
	const unsigned Dout = (Din+TopLeftPad+BottomRightPad);
	ap_uint<Cin*Ibit> temp_out = 0;

	for (unsigned rep = 0; rep < reps; rep++) {
#pragma HLS loop_tripcount min=1 max=1
		for (unsigned h = 0; h < TopLeftPad; h++) {
			for (unsigned s = 0; s < Dout; s++) {
				out.write(0);
			}
		}

		for (unsigned h = 0; h < Din; h++) { //800

			for ( unsigned s = 0; s < Dout; s++ ) {  //801
#pragma HLS PIPELINE II=1

				if ( (s < TopLeftPad) || (s >= Dout-BottomRightPad) ) {
					temp_out = 0;
				}
				else {
					temp_out = in.read();
				}

				out.write(temp_out);
			}
		}

		for (unsigned h = 0; h < BottomRightPad; h++) { //1
			for (unsigned i = 0; i < Dout; i++) { //801
				out.write(0);
			}
		}

	}
}


template <	unsigned K,
			unsigned S,
			unsigned Din,
			unsigned Cin,
			unsigned Ibit>
void SWU(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<Cin*Ibit> >& out,
	const unsigned reps)
{
	const unsigned steps = (Din-K)/S+1;  //400
	const unsigned line_buffer_size = K*Din;  //
#ifdef SWU_DEBUG
	cout << "steps: " << steps << endl;
	cout << "line_buffer_size: " << line_buffer_size << endl;
#endif

	ap_uint<Cin*Ibit> line_buffer[line_buffer_size];
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_2p
	//#pragma HLS RESOURCE variable line_buffer core=RAM_2P

	ap_uint<Cin*Ibit> temp_in;

	ap_uint<1> initial_fill = 0;
	unsigned stride = 0;
	unsigned pointer = 0;
	unsigned h = 0;

	for (unsigned rep = 0; rep < reps*Din; rep++) {  //801
#pragma HLS loop_tripcount min=Din max=Din
		if (h == Din) {
			initial_fill = 0;
			stride = 0;
			pointer = 0;
			h = 0;
		}
		h += 1;

#ifdef SWU_DEBUG
		cout << "wpointer: " << pointer << endl;
#endif

		for (unsigned w = 0; w < Din; w++) {  //801
#pragma HLS PIPELINE II=1
			temp_in = in.read();

			unsigned line_buffer_pointer = pointer + w;
			if (line_buffer_pointer >= line_buffer_size)
				line_buffer_pointer = line_buffer_pointer - line_buffer_size;
#ifdef SWU_DEBUG
			cout << "line_buffer_pointer: " << line_buffer_pointer << endl;
#endif
			line_buffer[line_buffer_pointer] = temp_in;
		}

		stride += 1;
		pointer += Din;
		if (pointer >= line_buffer_size) {
			pointer = pointer - line_buffer_size;
			initial_fill = 1;
#ifdef SWU_DEBUG
			cout << "initial_fill set to 1!" << endl;
#endif
		}

#ifdef SWU_DEBUG
		cout << "stride: " << stride << endl;
		cout << "rpointer: " << pointer << endl;
		cout << "line_buffer for out: ";
		for (unsigned j = 0; j < line_buffer_size; j++) {
			cout << line_buffer[j] << " ";
		}
		cout << endl;
#endif
		if (initial_fill == 1 && stride >= S) {
			stride = 0;

			unsigned s = 0;
			unsigned x = 0;
			unsigned y = 0;

			for (unsigned i = 0; i < steps*(K*K); i++ ) { //400*9 3600
#pragma HLS PIPELINE II=1

				unsigned read_address = (pointer+s*S) + y*Din + x;

				if (read_address >= line_buffer_size)
					read_address = read_address - line_buffer_size;
#ifdef SWU_DEBUG
				cout << "read_address: " << read_address << endl;
#endif
				ap_uint<Cin*Ibit> temp_out = line_buffer[read_address];
				out.write(temp_out);

				if (x == K-1) {
					x = 0;
					if (y == K-1) {
						y = 0;
						if (s == steps-1)
							s = 0;
						else
							s++;
					}
					else
						y++;
				}
				else
					x++;
			}


		}
	}
}




template <	unsigned K,
			unsigned Cin,
			unsigned Ibit>
void win_shift_left(ap_uint<Cin*Ibit>  win_kernel[K][K]){
#pragma HLS INLINE
	unsigned i,j;
	label1:for(i=0;i<K;i++){    //col
#pragma HLS UNROLL
			label11:for(j=0;j<K-1;j++){    //row
#pragma HLS UNROLL
				win_kernel[i][j]=win_kernel[i][j+1];
			}
	}
}

template <	unsigned K,
			unsigned Cin,
			unsigned Ibit,
			unsigned Din_W>
void win_from_buf(ap_uint<Cin*Ibit>  win_kernel[K][K],ap_uint<Cin*Ibit> line_buffer[K-1][Din_W],unsigned flag){
#pragma HLS INLINE
	unsigned i;
	label2:for(i=0;i<K-1;i++){    //col
#pragma HLS UNROLL
				win_kernel[i][K-1]=line_buffer[i][flag];
	}
}

template <	unsigned K,
			unsigned Cin,
			unsigned Ibit,
			unsigned Din_W>
void buf_shift_up(ap_uint<Cin*Ibit> line_buffer[K-1][Din_W],unsigned flag){
#pragma HLS INLINE
	unsigned i;
	label2:for(i=0;i<K-2;i++){    //col
#pragma HLS UNROLL
		      line_buffer[i][flag]=line_buffer[i+1][flag];
	}
}
/*
 * 2021_03_09 修正：使其支持K>3的情况
 * 2021/03/10 已经查证
 * 就是Rokcy中的SWU_NoWait
 */
template <	unsigned K,
			unsigned S,
			unsigned Din_W,  //进行Samepad后的中间宽居图为：IntermediateDout = S*(Dout-1) + K;
			unsigned Din_H,
			unsigned Cin,
			unsigned Ibit,
			unsigned InP>  //输入并行度 ；真实并行度为MVTU_InP*K*K  (MVTU_InP=Cin)
void SWU_KK(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<K*K*InP*Ibit> >& out,
	const unsigned reps = 1)
{
//	const unsigned parts = Cin/InP;
	ap_uint<Cin*Ibit> line_buffer[K-1][Din_W];
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_2p
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
//#pragma HLS RESOURCE variable line_buffer core=RAM_2P
	ap_uint<Cin*Ibit> kernel_win[K][K];
#pragma HLS ARRAY_PARTITION variable=kernel_win complete dim=0

	ap_uint<Cin*Ibit> temp_in;
	ap_uint<Cin*Ibit> win_reg1;
	ap_uint<Cin*Ibit> win_reg2;

	ap_uint<Cin*Ibit> temp_in_reg;

	ap_uint<InP*Ibit*K*K> temp;

	unsigned i,j,m,n;
for(unsigned rep=0;rep<reps;rep++){
#pragma HLS loop_tripcount min=1 max=1
		label2:for(i=0;i<Din_H;i++){
			label21:for(j=0;j<Din_W;j++){
	#pragma HLS PIPELINE II=1
                // win滑窗
				win_shift_left<K,Cin,Ibit>(kernel_win);
				// win从line_buffer取数
				win_from_buf<K,Cin,Ibit,Din_W>(kernel_win,line_buffer,j);
		//		kernel_win[0][2]=line_buffer[0][j];
		//		kernel_win[1][2]=line_buffer[1][j];
				temp_in = in.read();
				kernel_win[K-1][K-1]=temp_in;

				//linebuffer.shiftup()
				buf_shift_up<K,Cin,Ibit,Din_W>(line_buffer,j);
		//		line_buffer[0][j]=line_buffer[1][j];
	     		line_buffer[K-2][j]=temp_in;

				if(i>=K-1&&i%S==0&&j>=K-1&&j%S==0){
					for(m=0;m<K;m++){
				#pragma HLS UNROLL
						for(n=0;n<K;n++){
				#pragma HLS UNROLL
							temp_in_reg=kernel_win[m][n];
							temp((n+m*K+1)*InP*Ibit-1,(n+m*K)*InP*Ibit) = temp_in_reg;
			//				out.write(kernel_win[m][n]);
					}
				}
					out.write(temp);
				}
			}
		}
	}
}


/*
 * 2021_03_09 修正：使其支持K>3的情况
 * 2021/03/16 已经查证
 * 不仅支持深度可分离卷积Cin<=Inp,也支持Cin=Inp下K*K*Cin的标准卷积
 */
template <	unsigned K,
			unsigned S,
			unsigned Din_W,  //进行Samepad后的中间宽居图为：IntermediateDout = S*(Dout-1) + K;
			unsigned Din_H,
			unsigned Cin,
			unsigned Ibit,
			unsigned InP>  //输入并行度 ；真实并行度为MVTU_InP*K*K  (MVTU_InP=Cin)
void SWU_DSC(
	stream<ap_uint<Cin*Ibit> >& in,
	stream<ap_uint<K*K*InP*Ibit> >& out,
	const unsigned reps = 1)
{
	const unsigned parts = Cin/InP;
	ap_uint<Cin*Ibit> line_buffer[K-1][Din_W];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_2p
	ap_uint<Cin*Ibit> kernel_win[K][K];
#pragma HLS ARRAY_PARTITION variable=kernel_win complete dim=0

	ap_uint<Cin*Ibit> temp_in;
	ap_uint<Cin*Ibit> win_reg1;
	ap_uint<Cin*Ibit> win_reg2;

	ap_uint<Cin*Ibit> temp_in_reg;
//	ap_uint<Cin*Ibit> temp_in_reg;

	ap_uint<InP*Ibit> temp_out_reg;
	ap_uint<InP*Ibit*K*K> temp_out;

	unsigned i,j,m,n,loop;
for(unsigned rep=0;rep<reps;rep++){
#pragma HLS loop_tripcount min=1 max=1
		label2:for(i=0;i<Din_H;i++){
			label21:for(j=0;j<Din_W;j++){
#pragma HLS PIPELINE II=Cin/InP

                // win滑窗
				win_shift_left<K,Cin,Ibit>(kernel_win);
				// win从line_buffer取数
				win_from_buf<K,Cin,Ibit,Din_W>(kernel_win,line_buffer,j);
		//		kernel_win[0][2]=line_buffer[0][j];
		//		kernel_win[1][2]=line_buffer[1][j];
				temp_in = in.read();
				kernel_win[K-1][K-1]=temp_in;

				//linebuffer.shiftup()
				buf_shift_up<K,Cin,Ibit,Din_W>(line_buffer,j);
		//		line_buffer[0][j]=line_buffer[1][j];
	     		line_buffer[K-2][j]=temp_in;

				if(i>=K-1&&i%S==0&&j>=K-1&&j%S==0){
					SWU_DSC_label0:for(loop=0;loop<parts;loop++){
				#pragma HLS PIPELINE II=Cin/InP
						for(m=0;m<K;m++){
					#pragma HLS UNROLL
							for(n=0;n<K;n++){
					#pragma HLS UNROLL
								temp_in_reg=kernel_win[m][n];
								temp_out_reg=temp_in_reg((loop+1)*InP*Ibit-1,loop*InP*Ibit);
								temp_out((n+m*K+1)*InP*Ibit-1,(n+m*K)*InP*Ibit) = temp_out_reg;
				//				out.write(kernel_win[m][n]);
							}
						}
						out.write(temp_out);
					}

				}
			}
		}
	}
}

