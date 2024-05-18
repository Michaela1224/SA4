
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#define AP_INT_MAX_W 5000
#include "rocky-lib.h"
#include "config_sa.h"
#include "param.h"

#define MAX_OUT_COL 80

//#define PRINT_DEBUG

void ExtractPixels_var(
	stream<trans_pkt_in > & in,
	stream<ap_uint<256> >& out,
	const unsigned NumLines){
	trans_pkt_in temp;

	for (unsigned long long rep = 0; rep < NumLines; rep++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=NumLines max=NumLines
		temp = in.read();
		out.write( temp.data(256-1, 0) );
	}
}

void AddLast_var(
	stream<ap_uint<256> >& in,
	stream<trans_pkt_out >& out,
	const unsigned NumLines){
	trans_pkt_out temp;
	temp.keep = "0xf";

	for (unsigned long long i = 0; i < NumLines-1; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=NumLines-1 max=NumLines-1
		temp.data(256-1,0) = in.read();
		// temp.data(256-1,64)=0;
		temp.last = 0;
		out.write(temp);
	}

	temp.data(256-1,0) = in.read();
	// temp.data(256-1,64)=0;
	temp.last = 1;
	out.write(temp);
}




void W_IO_array(stream<ap_uint<4 * K * W_BIT> > fifo_W_local_out[MAX_A_ROW][MAX_A_COL],
                          const unsigned OUT_H,
                          const unsigned PENUM,
                          const unsigned SIMDNUM) {
#pragma HLS INLINE OFF

    ap_uint<MAX_INP * K *W_BIT>  w;
    ap_uint<4 * K *W_BIT>  temp;

#ifdef DEBUG
  FILE* fp_win0 = fopen("W3_reorg_SIMD_all_in.txt", "wb");
#endif


  for (unsigned int h = 0; h < OUT_H; h++) { // 40
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
        for (unsigned int infoldIdx = 0; infoldIdx < K* SIMDNUM; infoldIdx++) {
          for(unsigned int p = 0; p < 4; p++) {
#pragma HLS PIPELINE II=1
            for (unsigned int c = 0; c < MAX_A_COL; c++) {
                w=conv_all_w[c][peIdx*K*SIMDNUM*4+infoldIdx*4+p];
                for(unsigned int r = 0;  r< MAX_A_ROW; r++) {
                  temp=w((r+1)*4 * K *W_BIT-1,r*4 * K *W_BIT);
                  fifo_W_local_out[r][c].write(temp);
                }
            }
          }
        }
      }
    }

#ifdef DEBUG
  fclose(fp_win0);
#endif

}




void ExpandWidth_AXI(
	stream<ap_uint<MAX_IM_BIT*2> > in[MAX_OUP],
	stream<ap_uint<16*2*MAX_OUP> > & out,
	const unsigned NumLines){

    ap_uint<16*2*MAX_OUP> temp;
    ap_int<MAX_IM_BIT> reg0_Mbit,reg1_Mbit;
    ap_int<16> reg0_16bit;
    ap_int<16> reg1_16bit;
    // ap_uint<16*2> reg;

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400
#pragma HLS PIPELINE II=1
		for (unsigned p = 0; p < MAX_OUP; p++) {  //4
        (reg1_Mbit,reg0_Mbit)=in[p].read();
        reg0_16bit=reg0_Mbit;
        reg1_16bit=reg1_Mbit;
        temp((p+1)*16*2-1,p*16*2)=(reg1_16bit,reg0_16bit);
		}
		out.write(temp);
	}

}



void ReduceWidth_IN_Var(
	stream<ap_uint<256> > & in,
	stream<ap_uint<MAX_INP*2*4> > & out,
	const unsigned NumLines)
{

	const unsigned parts = 256/(MAX_INP*2*4);

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400*3*3
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1

		ap_uint<256> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<MAX_INP*2*4> temp_out = temp_in(MAX_INP*2*4-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> (MAX_INP*2*4);
		}
	}
}


template <	unsigned InStreamW,
			unsigned OutStreamW>
void ReduceWidth(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > & out,
	const unsigned NumLines)
{
	const unsigned parts = InStreamW/OutStreamW;

	for (unsigned rep = 0; rep < NumLines; rep++) {  //400*400*3*3
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=InStreamW/OutStreamW

		ap_uint<InStreamW> temp_in = in.read();
		for (unsigned p = 0; p < parts; p++) {

			ap_uint<OutStreamW> temp_out = temp_in(OutStreamW-1, 0);
			out.write( temp_out );
			temp_in = temp_in >> OutStreamW;
		}
	}
}


void convDSPOpt_SA(
    stream<ap_uint<MAX_INP * IN_BIT * 2> > &vec_in,
    stream<ap_uint<256> > &out) {

#pragma HLS DATAFLOW



 stream<ap_uint<MAX_INP * IN_BIT * 2> > vec_ori("vec_ori");
#pragma HLS STREAM variable=vec_ori depth=4


 stream<ap_uint<MAX_INP * IN_BIT * 2> > samepad_out("samepad_out");
#pragma HLS STREAM variable=samepad_out depth=4

stream<ap_uint<MAX_INP * IN_BIT * 2> > vec_pad("vec_pad");
#pragma HLS STREAM variable=vec_pad depth=16

stream<ap_uint< 4* IN_BIT * 2> > fifo_A_PE[MAX_A_ROW][MAX_A_COL];  //SIMD PE
#pragma HLS STREAM variable=fifo_A_PE depth=4

/* W_IO_L2_in fifo */ stream<ap_uint<4 * K * W_BIT> > fifo_W_PE[MAX_A_ROW][MAX_A_COL];
#pragma HLS STREAM variable=fifo_W_PE depth=8 dim=1
#pragma HLS STREAM variable=fifo_W_PE depth=8 dim=2


/* PE fifo */ stream<ap_uint<PROD_BIT*4> > fifo_C_PE[MAX_A_ROW][MAX_A_COL][4]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_PE type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_PE depth=16 dim=1
#pragma HLS STREAM variable=fifo_C_PE depth=16 dim=2
#pragma HLS STREAM variable=fifo_C_PE depth=16 dim=3
//#pragma HLS ARRAY_PARTITION variable=fifo_C_PE dim=0 complete


/* PE fifo */ stream<ap_uint<MAX_IM_BIT*2> > fifo_C_PE_ACC[MAX_A_ROW][MAX_A_COL][4]; // [SIMD][PE]
#pragma HLS BIND_STORAGE variable=fifo_C_PE_ACC type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_PE_ACC depth=4 dim=1
#pragma HLS STREAM variable=fifo_C_PE_ACC depth=4 dim=2
#pragma HLS STREAM variable=fifo_C_PE_ACC depth=4 dim=3
//#pragma HLS ARRAY_PARTITION variable=fifo_C_PE dim=0 complete


stream<ap_uint<MAX_IM_BIT*2> > fifo_C_PE_all[MAX_OUP];
#pragma HLS BIND_STORAGE variable=fifo_C_PE_all type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_PE_all depth=4 dim=1
//#pragma HLS STREAM variable=fifo_C_PE_all depth=4 dim=2
//#pragma HLS ARRAY_PARTITION variable=fifo_C_PE_all dim=0 complete

/* PE fifo */ stream<ap_uint<MAX_IM_BIT*2> > fifo_C_PE_reorg[MAX_OUP];
#pragma HLS BIND_STORAGE variable=fifo_C_PE_reorg type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_PE_reorg depth=4
//#pragma HLS ARRAY_PARTITION variable=fifo_C_PE_reorg dim=0 complete


/* PE fifo */ stream<ap_uint<16*2*MAX_OUP> > fifo_C_PE_res_exp;
#pragma HLS BIND_STORAGE variable=fifo_C_PE_res_exp type=fifo impl=autosrl
#pragma HLS STREAM variable=fifo_C_PE_res_exp depth=32
//#pragma HLS ARRAY_PARTITION variable=fifo_C_PE_res dim=0 complete

#ifdef FIRST_DEBUG // 40* (64/2) * 3 * (32/16)* (80/2) * (16*2*4 bit)
    cout << "vec.size(): " << vec.size() << endl;
#endif

const unsigned L_PENUM = L_OUT_CH / MAX_OUP;
const unsigned L_SIMDNUM = L_IN_CH / MAX_INP;

const unsigned numlines=L_OUT_H*L_PENUM*K*L_SIMDNUM*(L_IN_W/2);
const unsigned int total_num_out=L_OUT_H*L_PENUM*(L_IN_W/2);
const unsigned int L_OUT_W=L_IN_W; 

const unsigned L_Inter_H=L_IN_H+2;


#ifdef PRINT_DEBUG ////40*(64/2)* (80/2)*(32/16)*3*2=614400

    cout << "print necessary parameter when" << endl;
    cout <<"L_IN_H:"<<L_IN_H<<endl;
    cout <<"L_IN_W:"<<L_IN_W<<endl;
    cout <<"L_IN_CH:"<<L_IN_CH<<endl;
    cout <<"L_OUT_CH:"<<L_OUT_CH<<endl;


    cout <<"L_PENUM:"<<L_PENUM<<endl;
    cout <<"L_SIMDNUM:"<<L_SIMDNUM<<endl;


    cout <<"L_Inter_H:"<<L_Inter_H<<endl;
    cout <<"L_OUT_H:"<<L_OUT_H<<endl;

#endif



    SAMEPAD_DSPopt_SA_UP_DOWN<1,1,MAX_INP,IN_BIT>(vec_in, samepad_out, L_Inter_H, L_IN_W/2, L_IN_CH);


    conv3padding_opt_SA<3, IN_BIT, MAX_INP>(samepad_out, vec_pad,L_Inter_H,L_IN_W,L_OUT_H, L_IN_CH,L_PENUM);


#ifdef PRINT_DEBUG ////40*(64/2)* (80/2)*(32/16)*3*2=614400

    cout << "samepad+paading finish........" << endl;

#endif


#ifdef DEBUG
		ap_uint<MAX_INP * 4 * 2> tmp_0;
		ap_uint<4> tmp1_0;
        char fname[100];
        sprintf(fname,"padding_l%d_t%d.txt",layer+1,times);
		FILE* fp0 = fopen(fname, "wb");
		for(int i = 0;i < L_OUT_H*L_PENUM*K*L_SIMDNUM*(L_IN_W/2);i++){

            tmp_0=vec_pad.read();
            vec_pad.write(tmp_0);

			for(int j=0;j<16* 2;j++){      //out channel
				tmp1_0=tmp_0(4-1,0);        //out Abit
				tmp_0=tmp_0>>4;
				fprintf(fp0, "%lf\n", double(tmp1_0));
			}
		}
		fclose(fp0);
#endif

    A_to_array<MAX_A_ROW,MAX_A_COL,MAX_INP * IN_BIT * 2,4 * IN_BIT * 2>(
        vec_pad,
        fifo_A_PE,
        numlines
    );






    W_IO_array(fifo_W_PE,
               L_OUT_H,
               L_PENUM,
               L_SIMDNUM
    );




    for (unsigned int r=0; r< MAX_A_ROW;r++){  //有问题
    #pragma HLS UNROLL
        for (unsigned int c=0; c< MAX_A_COL;c++){
        #pragma HLS UNROLL

            // if(c==2){
            //   cout<<"debug:"<<endl;
            // }


            #ifdef DEBUG // 40* (64/2) * (80/4) * (32/16)*  (3*16)*(2bit)
                if(i==1 && j==7){
                cout << "start debug" << endl;
                // cout << "fifo_W_PE["<<i<<"]["<<j<<"]: "<< fifo_W_PE[i][j].size() << endl;
                // cout << "fifo_A_PE["<<i<<"]["<<j<<"]: "<< fifo_A_PE[i][j].size() << endl;
                // cout << "fifo_C_PE["<<i<<"]["<<j<<"]: "<< fifo_C_PE[i][j].size() << endl;
                }
            #endif

            #ifdef FIRST_DEBUG // 40* (64/2) * (80/4) * (32/16)*  (3*16)*(2bit)
                cout << "fifo_W_PE["<<r<<"]"<<"["<<c<<"]"<<"["<<i<<"]["<<j<<"]:" << fifo_W_PE[r][c][i][j].size() << endl;
                cout << "fifo_A_PE["<<r<<"]"<<"["<<c<<"]"<<"["<<i<<"]["<<j<<"]:" << fifo_A_PE[r][c][i][j].size() << endl;
                cout << "fifo_C_PE["<<j<<"]"<<"["<<i<<"]"<<"["<<r<<"]["<<c<<"]:" << fifo_C_PE[j][i][r][c].size() << endl;
            #endif

            PE_wrapper<K,IN_BIT,W_BIT,
                4,4,PROD_BIT>(
                /* module id */ r,  // PE
                /* module id */ c,  // SIMD
                /* fifo */ fifo_W_PE[r][c],
                /* fifo */ fifo_A_PE[r][c],
                /* fifo */ fifo_C_PE[r][c],
                L_IN_W,
                numlines
            );
            #ifdef FIRST_DEBUG // 40* (64/2) * (80/4) * (32/16)*  (3*16)*(2bit)
                cout << "fifo_A_PE["<<r<<"]"<<"["<<c<<"]"<<"["<<i+1<<"]["<<j<<"]:" << fifo_A_PE[r][c][i+1][j].size() << endl;
                cout << "fifo_C_PE["<<j+1<<"]"<<"["<<i<<"]"<<"["<<r<<"]["<<c<<"]:" << fifo_C_PE[j+1][i][r][c].size() << endl;
            #endif
            #ifdef FIRST_DEBUG
                cout<<"finish clear["<<r<<","<<c<<","<<i<<","<<j<< "]PE! "<<endl;
            #endif

            #ifdef PRINT_DEBUG
                cout<<"finish clear["<<r<<","<<c<<"]PE! "<<endl;
            #endif

        }
    }

//    cout<<"finish clear ALL PE! "<<endl;


// PE脉动阵列计算
    for (unsigned int r=0; r< MAX_A_ROW;r++){  //有问题
    #pragma HLS UNROLL
        for (unsigned int c=0; c< MAX_A_COL;c++){
        #pragma HLS UNROLL

            // if(c==2){
            //   cout<<"debug:"<<endl;
            // }

            PE_DSP_ACC<K,IN_BIT,W_BIT,
                4,4,PROD_BIT,MAX_IM_BIT>(
                /* module id */ r,  // PE
                /* module id */ c,  // SIMD
                /* fifo */ fifo_C_PE[r][c],
                /* fifo */ fifo_C_PE_ACC[r][c],
                L_OUT_H,
                L_IN_W,
                L_PENUM,
                L_SIMDNUM
            );
        }
    }

#ifdef Modify_DEBUG
    ap_uint<MAX_IM_BIT*2> tmp_C;
    ap_int<MAX_IM_BIT> tmp0_C;
    char fnameC[100];
    for (unsigned int r=0; r< MAX_A_ROW;r++){  //有问题
        for (unsigned int c=0; c< MAX_A_COL;c++){
            for(unsigned int m=0; m< 4;m++){
                sprintf(fnameC,"acc_r%d_c%d_pe%d.txt",r,c,m);
                FILE* fpC = fopen(fnameC, "wb");
                cout << "fifo_C_PE_ACC["<<r<<"]"<<"["<<c<<"]["<<m<<"]:" << fifo_C_PE_ACC[r][c][m].size() << endl;
                for(int i = 0;i < L_OUT_H*L_PENUM*K*L_SIMDNUM*(L_IN_W/2)+1;i++){
                    tmp_C=fifo_C_PE_ACC[r][c][m].read();
                    fifo_C_PE_ACC[r][c][m].write(tmp_C);
                    for(int j=0;j<2;j++){
                        tmp0_C=tmp_C((j+1)*MAX_IM_BIT-1,j*MAX_IM_BIT);
                        fprintf(fpC, "%lf\n", double(tmp0_C));
                    }
                }
                fclose(fpC);
            }
        }
    }
#endif



    arrar_acc_to_Res<MAX_A_ROW,MAX_A_COL,MAX_OUP,MAX_IM_BIT>(fifo_C_PE_ACC,fifo_C_PE_all,numlines);


    Inter_Reorg_acc_to_Res<K,MAX_IM_BIT, MAX_OUP>(fifo_C_PE_all,fifo_C_PE_reorg,L_OUT_H,L_IN_W, L_PENUM,L_SIMDNUM);


#ifdef DEBUG
    ap_int<MAX_IM_BIT> data_pe0_0,data_pe0_1;
    char fname_acc[100];
    sprintf(fname_acc,"acc_result_l%d_t%d.txt",layer+1,times);
    FILE* fp1 = fopen(fname_acc, "wb");
    for(int i = 0;i < (L_OUT_H*L_IN_W*L_OUT_CH/(16*2));i++){
        for(int j = 0;j < MAX_OUP;j++){
        (data_pe0_1,data_pe0_0)=fifo_C_PE_reorg[j].read();
        fifo_C_PE_reorg[j].write((data_pe0_1,data_pe0_0));
        fprintf(fp1, "%d\n", int(data_pe0_0));
        fprintf(fp1, "%d\n", int(data_pe0_1));
        }
    }
    fclose(fp1);
#endif




#ifdef DEBUG
    ap_uint<MAX_OUP * OUT_BIT * 2> tmp_PE;
    ap_uint<OUT_BIT> tmp2;
    char fname_act[100];
    sprintf(fname_act,"bn_result_l%d_t%d.txt",layer+1,times);
    FILE* fp_act = fopen(fname_act, "wb");
    for(int i = 0;i < total_num_out;i++){
        tmp_PE=out_in.read();
        out_in.write(tmp_PE);
        for(int k=0;k<MAX_OUP*2;k++){
            tmp2=tmp_PE(OUT_BIT-1,0);        //out Abit
            tmp_PE=tmp_PE>>OUT_BIT;
            fprintf(fp_act, "%d\n", int(tmp2));
        }
    }
    fclose(fp_act);
#endif


    ExpandWidth_AXI(fifo_C_PE_reorg,fifo_C_PE_res_exp,total_num_out);

    ReduceWidth<16*2*MAX_OUP,256>(fifo_C_PE_res_exp,out,total_num_out);


}





void sa_compute(stream<trans_pkt_in >& in,stream<trans_pkt_out >& out){
#pragma HLS INTERFACE s_axilite bundle=control port=return
#pragma HLS INTERFACE axis register_mode=both register port=in
#pragma HLS INTERFACE axis register_mode=both register port=out  // 20230509: see register_mode
//---------------------------------------参数切分-----------------------------------------

#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=conv_all_w dim=1 complete

    const unsigned int num_axi_in = L_IN_H * L_IN_W*L_IN_CH/(32*2);  
    // const unsigned int num_per_rep = L_IN_H * L_IN_W*L_IN_CH/(MAX_INP*2);  
    const unsigned int num_axi_out = L_IN_H * L_IN_W*L_IN_CH/(16); 

	stream<ap_uint<256> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable=in_stream_extract depth=64
	ExtractPixels_var(in,in_stream_extract,num_axi_in);

 stream<ap_uint<MAX_INP * IN_BIT * 2> > sa_in("sa_in");
#pragma HLS STREAM variable=sa_in depth=16

    ReduceWidth_IN_Var(in_stream_extract,sa_in,num_axi_in);

 stream<ap_uint<256> > sa_out("sa_out");
#pragma HLS STREAM variable=sa_out depth=64

    convDSPOpt_SA(sa_in, sa_out);

    AddLast_var(sa_out,out,num_axi_out);



}





