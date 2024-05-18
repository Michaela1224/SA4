#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include "misc.h"
#include "stream.h"
using namespace std;


#define print_debug
// #define CONV1_INDEBUG

template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned DATA_BIT,
			unsigned W_BIT
			>
ap_uint<OUT_BIT> BN_Re15( ap_int<IN_BIT> in,
                ap_int<INC_BIT> inc,
                ap_int<BIAS_BIT> bias ) {
 //#pragma HLS inline off
  const unsigned L_SHIFT=8;
	const unsigned D = 1 << (W_BIT - 1 + DATA_BIT + L_SHIFT);  //濠电偛顦崝宥夊礈閿燂拷
	const unsigned   Inter_BIT=IN_BIT+INC_BIT+1;
	ap_int<INC_BIT> inc_tmp=inc;
	ap_int<BIAS_BIT> bias_tmp=bias;
	ap_int<Inter_BIT> bn_res = in * inc_tmp + bias_tmp;
	ap_uint<OUT_BIT> res;
	ap_int<Inter_BIT+1> bn_res_tmp;
	if (bn_res > 0) {
		bn_res_tmp=(bn_res + (D >> 1)) ;
		bn_res_tmp = bn_res_tmp>> (W_BIT - 1 + DATA_BIT + L_SHIFT);   //闂佸憡鐗曢幖顐﹀春閿燂拷
		if (bn_res_tmp > 15){
			res = 15;
		} else {
			res = bn_res_tmp;
		}
	} else {
		res = 0;
	}
	return res;

}

/*
template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned INC_BIT,
			unsigned BIAS_BIT,
			unsigned DATA_BIT,
			unsigned W_BIT
			>
ap_uint<OUT_BIT> BN_Re15_LLSQ( ap_int<IN_BIT> in,
                ap_uint<INC_BIT> inc,
                ap_uint<BIAS_BIT> bias,
                const unsigned layer ) {
 //#pragma HLS inline off
  const unsigned L_SHIFT=L_SHIFT_array[layer];
	const unsigned D = 1 << (L_SHIFT);  //濠电偛顦崝宥夊礈閿燂拷
	const unsigned   Inter_BIT=IN_BIT+INC_BIT+1;
	ap_int<INC_BIT> inc_tmp=inc;
	ap_int<BIAS_BIT> bias_tmp=bias;
	ap_int<Inter_BIT> bn_res = (in+ bias_tmp)*inc_tmp;
  
  #ifdef out_DEBUG
    cout<<"inc: "<<inc_tmp<<endl;
    cout<<"input: "<<in<<endl;
    cout<<"bias: "<<in<<endl;
    cout<<"bias: "<<in<<endl;
  #endif 
  // ap_int<Inter_BIT> bn_res = in *  + bias_tmp;
	ap_uint<OUT_BIT> res;
	ap_int<Inter_BIT+1> bn_res_tmp;
	if (bn_res > 0) {
		bn_res_tmp=(bn_res + (D >> 1)) ;
		bn_res_tmp = bn_res_tmp>> (L_SHIFT);   //闂佸憡鐗曢幖顐﹀春閿燂拷
		if (bn_res_tmp > 15){
			res = 15;
		} else {
			res = bn_res_tmp;
		}
	} else {
		res = 0;
	}
	return res;

}

*/













template <unsigned IN_BIT, unsigned W_BIT>
ap_int<IN_BIT + W_BIT> conv_mul_lut(ap_uint<IN_BIT> in, ap_int<W_BIT> w) {
  ap_int<IN_BIT + W_BIT> out;
//#pragma HLS RESOURCE variable=return core=Mul_LUT
#pragma HLS inline off
  ap_int<IN_BIT> in_tmp;
  in_tmp=in;
  out = in_tmp * w;
  return out;
}




template <unsigned IN_BIT>
void loadInReg9(ap_uint<IN_BIT * 9> inData, ap_uint<IN_BIT> ivec[9]) {
#pragma HLS PIPELINE II=1
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable=ivec dim=1 complete

  for (unsigned s = 0; s < 9; s++) {
    ivec[s] = inData((s + 1) * IN_BIT - 1, s * IN_BIT);
  }
}

template <unsigned IN_BIT, unsigned W_BIT, unsigned PROD_BIT>
void simd_mac9_LUT(ap_uint<IN_BIT> invec[9], ap_int<W_BIT> w0vec[9],
                    ap_int<W_BIT> w1vec[9], ap_int<PROD_BIT> &out0,
                    ap_int<PROD_BIT> &out1) {
#pragma HLS ARRAY_PARTITION variable=invec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=w1vec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=w0vec dim=1 complete

  ap_int<PROD_BIT> acc0 = 0;
  ap_int<PROD_BIT> acc1 = 0;

  for (int i = 0; i < 9; i++) {
    ap_int<IN_BIT + W_BIT> m0 = conv_mul_lut<IN_BIT, W_BIT>(invec[i], w0vec[i]);
    acc0 += m0;
  }

  for (int i = 0; i < 9; i++) {
    ap_int<IN_BIT + W_BIT> m1 = conv_mul_lut<IN_BIT, W_BIT>(invec[i], w1vec[i]);
    acc1 += m1;
  }

  out0 = acc0;
  out1 = acc1;
}

template <	unsigned OUT_ROW,
			unsigned OUT_COL,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned IM,
			unsigned MVTU,
			unsigned InP,
			unsigned OutP>
void MVAU_l0_DSP(stream<ap_uint<Ibit * 9 * 3> > &in,
                   const ap_uint<3 * Wbit> weights[OutP][9][(Cout / OutP)],  // 权重存放与分割
                   stream<ap_uint<IM * OutP> > &out, const unsigned reps = 1) {

#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2

  const unsigned PROD_BIT = Ibit + Wbit + 4;

  ap_int<IM> outPartialArr[OutP]; 
#pragma HLS ARRAY_PARTITION variable=outPartialArr dim=1 complete

	unsigned wMat0 = 0;
	unsigned wMat1 = 0;

  	ap_uint<OutP*IM> resultVec;

	const unsigned long long totalReps = reps*OUT_ROW* (Cout/OutP)*OUT_COL;

	for (unsigned long long rep = 0; rep <totalReps; rep++) {  //50*50
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=OUT_ROW*(Cout/OutP)*OUT_COL max=OUT_ROW*(Cout/OutP)*OUT_COL
        ap_uint<Ibit> ivec[9];
#pragma HLS ARRAY_PARTITION variable=ivec dim=1 complete
        ap_uint<Ibit> ivec1[9];
#pragma HLS ARRAY_PARTITION variable=ivec1 dim=1 complete
        ap_uint<Ibit> ivec2[9];
#pragma HLS ARRAY_PARTITION variable=ivec2 dim=1 complete

        ap_int<Wbit> wvec[OutP][9];
#pragma HLS ARRAY_PARTITION variable=wvec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wvec dim=2 complete
        ap_int<Wbit> wvec1[OutP][9];
#pragma HLS ARRAY_PARTITION variable=wvec1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wvec1 dim=2 complete
        ap_int<Wbit> wvec2[OutP][9];
#pragma HLS ARRAY_PARTITION variable=wvec2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wvec2 dim=2 complete

        ap_uint<Ibit * 9> inData, inData1, inData2;
        (inData2, inData1, inData) = in.read();
        loadInReg9<Ibit>(inData, ivec);
        loadInReg9<Ibit>(inData1, ivec1);
        loadInReg9<Ibit>(inData2, ivec2);

        #ifdef INDEBUG
          cout<<"input act begin-------"<<endl;
          for(int i=0;i<9;i++){
            ap_int<8> tmp=ivec[i];
            cout<<tmp<<endl;
          }
          for(int i=0;i<9;i++){
            ap_int<8> tmp=ivec1[i];
            cout<<tmp<<endl;
          }
          for(int i=0;i<9;i++){
            ap_int<8> tmp=ivec2[i];
            cout<<tmp<<endl;
          }
          cout<<"input act end-------"<<endl;
        #endif

        for (int i = 0; i < OutP; i++) { // 8    // 同时获取，所以展开  // w从0-OUT_COL-1不动
          for (int s = 0; s < 9; s++) {
            wvec[i][s] = weights[i][s / 3][wMat1](        // s / 3
                (s % 3 + 1) * Wbit - 1, s % 3 * Wbit);
            wvec1[i][s] = weights[i][(s / 3) + 3][wMat1]( // (s / 3) + 3
                (s % 3 + 1) * Wbit - 1, s % 3 * Wbit);
            wvec2[i][s] = weights[i][(s / 3) + 6][wMat1]( // (s / 3) + 6
                (s % 3 + 1) * Wbit - 1, s % 3 * Wbit);
          }
        }

        #ifdef INDEBUG
          cout<<"input w begin-------"<<endl;
          for(int i=0;i<9;i++){
            ap_int<4> tmp=wvec[0][i];
            cout<<tmp<<endl;
          }
          for(int i=0;i<9;i++){
            ap_int<4> tmp=wvec1[0][i];
            cout<<tmp<<endl;
          }
          for(int i=0;i<9;i++){
            ap_int<4> tmp=wvec2[0][i];
            cout<<tmp<<endl;
          }
          cout<<"input w end-------"<<endl;
        #endif

        for (int p = 0; p < OutP; p += 2) { // 注意这里PE/2 unroll 为PE/2份
          ap_int<PROD_BIT> outPartial00;
          ap_int<PROD_BIT> outPartial01;
          ap_int<PROD_BIT> outPartial10;
          ap_int<PROD_BIT> outPartial11;
          ap_int<PROD_BIT> outPartial20;
          ap_int<PROD_BIT> outPartial21;

          simd_mac9_LUT<Ibit, Wbit, PROD_BIT>(ivec, wvec[p], wvec[p + 1],
                                                  outPartial00, outPartial01);
          simd_mac9_LUT<Ibit, Wbit, PROD_BIT>(ivec1, wvec1[p], wvec1[p + 1],
                                                  outPartial10, outPartial11);
          simd_mac9_LUT<Ibit, Wbit, PROD_BIT>(ivec2, wvec2[p], wvec2[p + 1],
                                                  outPartial20, outPartial21);   

          outPartialArr[p] = outPartial00 + outPartial10 + outPartial20;
          outPartialArr[p + 1] = outPartial01 + outPartial11 +  outPartial21;
        }


		// outPartialArr_tmp=outPartialArr;



        ap_uint<IM * OutP> odata;

        for (int i = 0; i < OutP; i++) {
          odata((i + 1) * IM - 1, i * IM) = outPartialArr[i];
        }

		out.write(odata);


		if (wMat0 == OUT_COL-1) {
			wMat0=0;
			if(wMat1 == Cout / OutP -1)
				wMat1=0;
			else
				wMat1++;
		}
		else
				wMat0++;

    }
}



template <	unsigned OUT_ROW,
			unsigned OUT_COL,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned IM,
			unsigned MVTU,
			unsigned InP,
			unsigned OutP>
void MVAU_l0_DSP_1K(stream<ap_uint<Ibit * 3 * 3> > &in,
                   const ap_uint<3 *3* Wbit> weights[OutP][3*(Cout / OutP)],  // 权重存放与分割
                   stream<ap_uint<IM * OutP> > &out, const unsigned reps = 1) {

#pragma HLS ARRAY_PARTITION variable=weights complete dim=1

  const unsigned PROD_BIT = Ibit + Wbit + 4;

  ap_int<IM> outPartialArr[OutP]; 
#pragma HLS ARRAY_PARTITION variable=outPartialArr dim=1 complete
  ap_int<PROD_BIT> outPartial0;
  ap_int<PROD_BIT> outPartial1;
  ap_uint<IM * OutP> odata;
	unsigned wMat0 = 0;
	unsigned wMat1 = 0;
  unsigned wMatk = 0;


	const unsigned long long totalReps = reps*OUT_ROW* (Cout/OutP)*OUT_COL*3;

  	ap_uint<OutP*IM> resultVec;
    ap_uint<Ibit> ivec[9];
#pragma HLS ARRAY_PARTITION variable=ivec dim=1 complete
    ap_int<Wbit> wvec[OutP][9];
#pragma HLS ARRAY_PARTITION variable=wvec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wvec dim=2 complete
    ap_uint<Ibit * 9> inData;

	for (unsigned long long rep = 0; rep <totalReps; rep++) {  //50*50
#pragma HLS PIPELINE II=1
#pragma HLS loop_tripcount min=OUT_ROW*(Cout/OutP)*OUT_COL*3 max=OUT_ROW*(Cout/OutP)*OUT_COL*3

        
        inData = in.read();
        loadInReg9<Ibit>(inData, ivec);  // ivec 先输入通道；3个后一行窗口的3个

        #ifdef INDEBUG
          cout<<"input act begin-------"<<endl;
          for(int i=0;i<9;i++){
            ap_int<8> tmp=ivec[i];
            cout<<tmp<<endl;
          }
          cout<<"input act end-------"<<endl;
        #endif

        for (int i = 0; i < OutP; i++) { // 8    // 同时获取，所以展开  // w从0-OUT_COL-1不动
          for (int s = 0; s < 9; s++) {
            wvec[i][s] = weights[i][wMat1*3+wMatk]((s+ 1) * Wbit - 1, s* Wbit);
          }
        }

        #ifdef INDEBUG
          cout<<"input w begin-------"<<endl;
          for(int i=0;i<9;i++){
            ap_int<4> tmp=wvec[0][i];
            cout<<tmp<<endl;
          }
          cout<<"input w end-------"<<endl;
        #endif

        for (int p = 0; p < OutP; p += 2) { // 注意这里PE/2 unroll 为PE/2份


          simd_mac9_LUT<Ibit, Wbit, PROD_BIT>(ivec, wvec[p], wvec[p + 1],
                                                  outPartial0, outPartial1);
          if(wMatk==0){
            outPartialArr[p] = outPartial0;
            outPartialArr[p + 1] = outPartial1;
          }
          else{
            outPartialArr[p] = outPartialArr[p]+outPartial0;
            outPartialArr[p + 1] = outPartialArr[p + 1]+outPartial1;            
          }

        }
		// outPartialArr_tmp=outPartialArr;
      if(wMatk==2){
        for (int i = 0; i < OutP; i++) {
          odata((i + 1) * IM - 1, i * IM) = outPartialArr[i];
        }

		    out.write(odata);
      }



    if(wMatk==2){
      wMatk=0;
      if (wMat0 == OUT_COL-1) {
        wMat0=0;
        if(wMat1 == Cout / OutP -1)
          wMat1=0;
        else
          wMat1++;
      }
      else
          wMat0++;
      }
    else{
      wMatk++;
    }
  }
}


template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS ARRAY_PARTITION variable=ipack dim=1 complete

  for (int i = 0; i < SIMD; i++) {
    ipack[i] =
        (A(i * IN_BIT + IN_BIT - 1, i * IN_BIT), (ap_uint<PROD_BIT - IN_BIT>)0,
         B(i * IN_BIT + IN_BIT - 1, i * IN_BIT));
  }
}


template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data_1K(ap_uint<2* IN_BIT*SIMD> A,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD]) {
#pragma HLS ARRAY_PARTITION variable=ipack dim=1 complete
  ap_uint<IN_BIT> A0,A1;
  for (int i = 0; i < SIMD; i++) {
    // (A1,A0)=A(2*i * IN_BIT + 2* IN_BIT - 1, 2*i * IN_BIT);
    (A1,A0)=A((i+1)*2* IN_BIT - 1, 2*i * IN_BIT);
    ipack[i] = (A1, (ap_uint<PROD_BIT - IN_BIT>)0,A0);
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS ARRAY_PARTITION variable=ipack dim=1 complete

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data_1K(ap_uint<3*W_BIT * SIMD> w,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS ARRAY_PARTITION variable=ipack dim=1 complete
  ap_int<W_BIT> w2_seg,w1_seg,w0_seg;
  for (int i = 0; i < SIMD; i++) {
    // (w2_seg,w1_seg,w0_seg)=w((3*i+1 * W_BIT + 3*W_BIT - 1, 3*i * W_BIT);
    (w2_seg,w1_seg,w0_seg)=w((i+1)*3*W_BIT - 1, 3*i * W_BIT);

    wpack[i] = (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE>
void simd_MAC(ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {
#pragma HLS ARRAY_PARTITION variable=wpack dim=1 complete
#pragma HLS ARRAY_PARTITION variable=ipack dim=1 complete
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  
  for (int i = 0; i < SIMD; i += CASCADE) {
#pragma HLS PIPELINE II=1
#pragma HLS unroll 
    ap_int<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
#pragma HLS unroll
      m += wpack[i + cs] * ipack[i + cs];


  #ifdef OutMVAU_DEBUG
    if(i==4&&cs==2){
      cout<<"wpack:"<<wpack[i + cs].to_string(2).c_str()<< '\n';
      cout<<"ipack:"<<ipack[i + cs].to_string(2).c_str()<< '\n';
      cout<<"m:"<<m.to_string(2).c_str()<< '\n';
      ap_int<PROD_BIT + 1>  p2_test = m(PROD_BIT * 3 - 1, PROD_BIT * 2 - 1);
      cout<<"m:"<<p2_test.to_string(2).c_str()<< '\n';
    }
  #endif

    }

    ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_int<PROD_BIT + 1> p1 = m(PROD_BIT * 2 - 1, PROD_BIT - 1);
    ap_int<PROD_BIT + 1> p2 = m(PROD_BIT * 3 - 1, PROD_BIT * 2 - 1);
    ap_int<PROD_BIT + 1> p3 = m(PROD_BIT * 4 - 1, PROD_BIT * 3 - 1);

  #ifdef OutMVAU_DEBUG

    if(i==4){
      cout<<"r2:"<<r2.to_string(2).c_str()<<endl;
      cout<<"r2:"<<r2<<endl;
    }
  #endif

    r0 += p0;
    r1 += (p1 >> 1) + (p1 & 1);
    r2 += (p2 >> 1) + (p2 & 1);
  #ifdef OutMVAU_DEBUG

    if(i==4){
      cout<<"p2:"<<p2.to_string(2).c_str()<< '\n';
      cout<<"r2:"<<r2.to_string(2).c_str()<<endl;
      cout<<"r2:"<<r2<<endl;
    }
  #endif
    r3 += (p3 >> 1) + (p3 & 1);

  }

  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;

  #ifdef OutMVAU_DEBUG
    cout<<"partial2:"<<partial2<<endl;
  #endif
}








template <	
			unsigned OUT_H,
			unsigned OUT_W, 
			unsigned K,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned IM,
			unsigned MVTU,
			unsigned InP,
			unsigned OutP,
			unsigned GUARD_BIT,
			unsigned CASCADE>
void MVAU_DSPOpt(
	stream<ap_uint<InP*Ibit*2*K> >& vec,
	const ap_uint<InP*Wbit> weights[OutP][K*K][(Cin/InP)*(Cout/OutP)],
	stream<ap_uint<MVTU*IM*2> >& out,
	const unsigned reps = 1){

	const unsigned PENUM = Cout/OutP;
	const unsigned SIMDNUM = Cin/InP;
	const unsigned PROD_BIT = Wbit + Ibit + GUARD_BIT;
	const unsigned WPACK_BIT = Wbit * 3 + Ibit * 2 + GUARD_BIT * 2;
	const unsigned IPACK_BIT = Ibit * 2 + Wbit + GUARD_BIT * 1;
	const unsigned OUT_WNUM = OUT_W / 2;
	const unsigned CycleNum= OutP/MVTU;

#pragma HLS ARRAY_PARTITION variable=weights dim=1 complete
#pragma HLS ARRAY_PARTITION variable=weights dim=2 complete

	ap_int<WPACK_BIT> wpacks0[OutP][InP];
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=2 complete
	ap_int<WPACK_BIT> wpacks1[OutP][InP];
#pragma HLS ARRAY_PARTITION variable=wpacks1 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wpacks1 dim=2 complete
	ap_int<WPACK_BIT> wpacks2[OutP][InP];
#pragma HLS ARRAY_PARTITION variable=wpacks2 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wpacks2 dim=2 complete

	ap_uint<IPACK_BIT> ipack0[InP];
#pragma HLS ARRAY_PARTITION variable=ipack0 dim=1 complete
	ap_uint<IPACK_BIT> ipack1[InP];
#pragma HLS ARRAY_PARTITION variable=ipack1 dim=1 complete
	ap_uint<IPACK_BIT> ipack2[InP];
#pragma HLS ARRAY_PARTITION variable=ipack2 dim=1 complete

	ap_int<IM> firPartialRes0[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes0 dim=1 complete
	ap_int<IM> firPartialRes1[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes1 dim=1 complete

	ap_int<IM> outPartialArr0[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr0 dim=1 complete
	ap_int<IM> outPartialArr1[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr1 dim=1 complete

	ap_int<PROD_BIT + 5> firPartial00;
	ap_int<PROD_BIT + 5> firPartial01;
	ap_int<PROD_BIT + 5> firPartial02;
	ap_int<PROD_BIT + 5> firPartial03;

	ap_int<PROD_BIT + 5> firPartial10;
	ap_int<PROD_BIT + 5> firPartial11;
	ap_int<PROD_BIT + 5> firPartial12;
	ap_int<PROD_BIT + 5> firPartial13;

	ap_int<PROD_BIT + 5> firPartial20;
	ap_int<PROD_BIT + 5> firPartial21;
	ap_int<PROD_BIT + 5> firPartial22;
	ap_int<PROD_BIT + 5> firPartial23;

	unsigned x_simdnum = 0;
	unsigned x_outwnum = 0;
	unsigned x_penum = 0;

	ap_uint<MVTU * IM> out_buf0;
	ap_uint<MVTU * IM> out_buf1;

	ap_uint<OutP * IM> out_buf0_tmp;
	ap_uint<OutP * IM> out_buf1_tmp;

	const unsigned long long totalReps = reps*OUT_H*PENUM*OUT_WNUM*SIMDNUM;


	for (unsigned long long rep = 0; rep <totalReps; rep++) {  //50*50
#pragma HLS loop_tripcount min=OUT_H*PENUM*OUT_WNUM*SIMDNUM max=OUT_H*PENUM*OUT_WNUM*SIMDNUM
#pragma HLS PIPELINE II=1

	    bool m_clear = (x_outwnum == 0);
        bool o_clear = (x_simdnum == 0);
        bool o_out = (x_simdnum == SIMDNUM - 1);
	
          ap_uint<InP * Ibit> data1[K], data0[K];  // 拿 3*SIMD
#pragma HLS ARRAY_PARTITION variable=data0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data1 dim=1 complete
          (data1[0], data0[0], data1[1], data0[1], data1[2], data0[2]) = vec.read();
          pack_input_data<Ibit, InP, PROD_BIT>(data1[0], data0[0], ipack0);
          pack_input_data<Ibit, InP, PROD_BIT>(data1[1], data0[1], ipack1);
          pack_input_data<Ibit, InP, PROD_BIT>(data1[2], data0[2], ipack2);

        #ifdef CONV1_INDEBUG
          cout<<"input act begin-------"<<endl;
          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data0[0](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }

          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data1[0](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }
          cout<<"first group end-------"<<endl;

          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data0[1](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }

          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data1[1](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }
          cout<<"second group end-------"<<endl;

          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data0[2](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }

          for(int i=0;i< InP;i++){
            ap_uint<4> tmp=data1[2](i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }
          cout<<"third group end-------"<<endl;

          cout<<"input act end-------"<<endl;
        #endif

          for (unsigned p = 0; p < OutP; p++) {
#pragma HLS unroll
            pack_weight_data<Wbit, InP, PROD_BIT>(
                weights[p][2][x_penum * SIMDNUM + x_simdnum],
                weights[p][1][x_penum * SIMDNUM + x_simdnum],
                weights[p][0][x_penum * SIMDNUM + x_simdnum], wpacks0[p]);
            pack_weight_data<Wbit, InP, PROD_BIT>(
                weights[p][5][x_penum * SIMDNUM + x_simdnum],
                weights[p][4][x_penum * SIMDNUM + x_simdnum],
                weights[p][3][x_penum * SIMDNUM + x_simdnum], wpacks1[p]);
            pack_weight_data<Wbit, InP, PROD_BIT>(
                weights[p][8][x_penum * SIMDNUM + x_simdnum],
                weights[p][7][x_penum * SIMDNUM + x_simdnum],
                weights[p][6][x_penum * SIMDNUM + x_simdnum], wpacks2[p]);
          }

        #ifdef CONV1_INDEBUG
        ap_uint<Wbit * InP> w2,w1,w0;
        cout<<"input w begin-------"<<endl;
        w2= weights[0][2][1];
        w1= weights[0][1][1];
        w0= weights[0][0][1];
        for (int i = 0; i < InP; i++) {
            ap_int<Wbit> w2_seg = w2(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w1_seg = w1(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w0_seg = w0(i * 4 + 4 - 1, i * 4);
            cout<<w0_seg<<endl;
            cout<<w1_seg<<endl;
            cout<<w2_seg<<endl;
        }
        cout<<"first group end-------"<<endl;
        w2= weights[0][5][1];
        w1= weights[0][4][1];
        w0= weights[0][3][1];

        for (int i = 0; i < InP; i++) {
            ap_int<Wbit> w2_seg = w2(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w1_seg = w1(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w0_seg = w0(i * 4 + 4 - 1, i * 4);
            cout<<w2_seg<<endl;
            cout<<w1_seg<<endl;
            cout<<w0_seg<<endl;
        }
        cout<<"second group end-------"<<endl;

        w2= weights[0][8][1];
        w1= weights[0][7][1];
        w0= weights[0][6][1];

        for (int i = 0; i < InP; i++) {
            ap_int<Wbit> w2_seg = w2(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w1_seg = w1(i * 4 + 4 - 1, i * 4);
            ap_int<Wbit> w0_seg = w0(i * 4 + 4 - 1, i * 4);
            cout<<w2_seg<<endl;
            cout<<w1_seg<<endl;
            cout<<w0_seg<<endl;
        }
        cout<<"third group end-------"<<endl;

          cout<<"input w end-------"<<endl;
        #endif


          for (int p = 0; p < OutP; p++) {
            // cout << "FIR result compare " << endl;
#pragma HLS unroll 

            simd_MAC<Wbit, Ibit, PROD_BIT, InP, CASCADE>(
                wpacks0[p], ipack0, firPartial00, firPartial01, firPartial02,
                firPartial03);
            
            simd_MAC<Wbit, Ibit, PROD_BIT, InP, CASCADE>(
                wpacks1[p], ipack1, firPartial10, firPartial11, firPartial12,
                firPartial13);
            

            #ifdef OutMVAU_DEBUG
              if (p==9&&rep==2){
                  cout<<"start debug:"<<firPartial00<<endl;
              }
            #endif

            simd_MAC<Wbit, Ibit, PROD_BIT, InP, CASCADE>(
                wpacks2[p], ipack2, firPartial20, firPartial21, firPartial22,
                firPartial23);
            // getchar();
            if (m_clear & o_clear) {  // 1 1
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial01 + firPartial11 + firPartial21;
            }
            if (m_clear & !o_clear) { // 1 0
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial01 + firPartial11 + firPartial21;
            } 
            if (!m_clear & o_clear) {// 0 1
              outPartialArr0[p] = firPartial00 + firPartial10 + firPartial20 + firPartialRes0[p];  //debug

            #ifdef DEBUG
              if (p==9){
                  cout<<"firPartialRes0[9]:"<<firPartialRes0[p]<<endl;
                  cout<<"firPartial00:"<<firPartial00<<endl;
                  cout<<"firPartial10:"<<firPartial10<<endl;
                  cout<<"firPartial20:"<<firPartial20<<endl;
              }

            #endif

              outPartialArr1[p] = firPartial01 + firPartial11 + firPartial21 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {// 0 0     
              outPartialArr0[p] += firPartial00 + firPartial10 + firPartial20;   // debug

            #ifdef DEBUG
              if (p==9){
                  cout<<"firPartial00:"<<firPartial00<<endl;
                  cout<<"firPartial10:"<<firPartial10<<endl;
                  cout<<"firPartial20:"<<firPartial20<<endl;
              }

            #endif

              outPartialArr1[p] += firPartial01 + firPartial11 + firPartial21;
            }

            if (o_clear) { //1 // 缓存上一步的结果
              firPartialRes0[p] = firPartial02 + firPartial12 + firPartial22;  // P2 //这是直接等于  # debug
            // #ifdef OutMVAU_DEBUG
            //   if (p==9){
            //       cout<<"firPartial02:"<<firPartial02<<endl;
            //       cout<<"firPartial12:"<<firPartial12<<endl;
            //       cout<<"firPartial22:"<<firPartial22<<endl;
            //   }

            // #endif

              firPartialRes1[p] = firPartial03 + firPartial13 + firPartial23; // P3
            }
            else {
              firPartialRes0[p] += firPartial02 + firPartial12 + firPartial22; //P2    // 注意这是累加

            #ifdef OutMVAU_DEBUG
              if (p==9&&x_simdnum==2){
                  cout<<"firPartial02:"<<firPartial02<<endl;
                  cout<<"firPartial12:"<<firPartial12<<endl;
                  cout<<"firPartial22:"<<firPartial22<<endl;
              }

            #endif

              firPartialRes1[p] += firPartial03 + firPartial13 + firPartial23;  // P3
            }

          }	

		if(rep>=SIMDNUM & x_simdnum<CycleNum){
			out_buf0=out_buf0_tmp(MVTU*IM-1,0);
			out_buf1=out_buf1_tmp(MVTU*IM-1,0);
			out.write((out_buf1, out_buf0));
			out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
			out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
		}
		
          if (o_out) {
            for (int p = 0; p < OutP; p++) {
#pragma HLS unroll 
              out_buf0_tmp(p * IM + IM - 1, p * IM) = outPartialArr0[p];
              out_buf1_tmp(p * IM + IM - 1, p * IM) = outPartialArr1[p];
            }
            #ifdef OutMVAU_DEBUG
              cout<<outPartialArr0[9]<<endl;
            #endif
		      }



		if (x_simdnum == SIMDNUM-1) {
			x_simdnum=0;
			if(x_outwnum == OUT_WNUM -1){
				x_outwnum=0;
				if(x_penum == PENUM-1){
					x_penum=0;
				}
				else{
					x_penum++;
				}
			}
			else{
				x_outwnum++;
			}
		}
		else{
			x_simdnum++;
		}

	
	
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out_buf1=out_buf1_tmp(MVTU*IM-1,0);
		out.write((out_buf1, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
		out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
	}


	for (int p = 0; p < OutP; p++) {
	#pragma HLS unroll
		out_buf0_tmp(p * IM + IM - 1, p * IM) = firPartialRes0[p];
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out.write((0, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
	}

}




template <	
			unsigned OUT_H,
			unsigned OUT_W, 
			unsigned K,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned IM,
			unsigned MVTU,
			unsigned InP,
			unsigned OutP,
			unsigned GUARD_BIT,
			unsigned CASCADE>
void MVAU_DSPOpt_1K(
	stream<ap_uint<InP*Ibit*2> >& vec,
	const ap_uint<InP*K*Wbit> weights[OutP][(Cin*K/InP)*(Cout/OutP)],
	stream<ap_uint<MVTU*IM*2> >& out,
	const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable=weights dim=1 complete
	const unsigned PENUM = Cout/OutP;
	const unsigned SIMDNUM = K*Cin/InP;
	const unsigned PROD_BIT = Wbit + Ibit + GUARD_BIT;
	const unsigned WPACK_BIT = Wbit * 3 + Ibit * 2 + GUARD_BIT * 2;
	const unsigned IPACK_BIT = Ibit * 2 + Wbit + GUARD_BIT * 1;
	const unsigned OUT_WNUM = OUT_W / 2;
	const unsigned  CycleNum= OutP/MVTU;




	ap_int<WPACK_BIT> wpacks[OutP][InP];
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=2 complete

	ap_uint<IPACK_BIT> ipack[InP];
#pragma HLS ARRAY_PARTITION variable=ipack0 dim=1 complete


	ap_int<IM> firPartialRes0[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes0 dim=1 complete
	ap_int<IM> firPartialRes1[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes1 dim=1 complete

	ap_int<IM> outPartialArr0[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr0 dim=1 complete
	ap_int<IM> outPartialArr1[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr1 dim=1 complete

	ap_int<PROD_BIT + 5> firPartial0;
	ap_int<PROD_BIT + 5> firPartial1;
	ap_int<PROD_BIT + 5> firPartial2;
	ap_int<PROD_BIT + 5> firPartial3;


	unsigned x_simdnum = 0;
	unsigned x_outwnum = 0;
	unsigned x_penum = 0;

	ap_uint<MVTU * IM> out_buf0;
	ap_uint<MVTU * IM> out_buf1;

	ap_uint<OutP * IM> out_buf0_tmp;
	ap_uint<OutP * IM> out_buf1_tmp;

  ap_uint<InP *2* Ibit> data;  // 拿 3*SIMD

	const unsigned long long totalReps = reps*OUT_H*PENUM*OUT_WNUM*SIMDNUM;


	for (unsigned long long rep = 0; rep <totalReps; rep++) {  //50*50
#pragma HLS loop_tripcount min=OUT_H*PENUM*OUT_WNUM*SIMDNUM max=OUT_H*PENUM*OUT_WNUM*SIMDNUM
#pragma HLS PIPELINE II=1

	    bool m_clear = (x_outwnum == 0);
      bool o_clear = (x_simdnum == 0);
      bool o_out = (x_simdnum == SIMDNUM - 1);
	
          
#pragma HLS ARRAY_PARTITION variable=data0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data1 dim=1 complete
          data = vec.read();
          pack_input_data_1K<Ibit, InP, PROD_BIT>(data, ipack);


        #ifdef INDEBUG
          cout<<"input act begin-------"<<endl;
          for(int i=0;i< 2*InP;i++){
            ap_uint<4> tmp=data(i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }
          cout<<"input act end-------"<<endl;

          cout<<"input pack act begin-------"<<endl;
          for(int i=0;i< InP;i++){
            cout<<ipack[i]<<endl;
          }
          cout<<"input pack act end-------"<<endl;

        #endif

          for (unsigned p = 0; p < OutP; p++) {
#pragma HLS unroll
            pack_weight_data_1K<Wbit, InP, PROD_BIT>(weights[p][x_penum * SIMDNUM + x_simdnum], wpacks[p]);
          }

        #ifdef INDEBUG
        ap_uint<3*Wbit * InP> w;
        ap_int<Wbit> w2_seg ;
        ap_int<Wbit> w1_seg ;
        ap_int<Wbit> w0_seg ;
        cout<<"input w begin-------"<<endl;
        w= weights[0][x_penum * SIMDNUM + x_simdnum];
        for (int i = 0; i < InP; i++) {
            (w2_seg,w1_seg,w0_seg)=w((i+1)*3*Wbit - 1, 3*i * Wbit);
            cout<<w0_seg<<endl;
            cout<<w1_seg<<endl;
            cout<<w2_seg<<endl;
        }
        cout<<"input w end-------"<<endl;
        #endif


          for (int p = 0; p < OutP; p++) {
            // cout << "FIR result compare " << endl;
#pragma HLS unroll 

            simd_MAC<Wbit, Ibit, PROD_BIT, InP, CASCADE>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2,
                firPartial3);
            

            // getchar();
            if (m_clear & o_clear) {  // 1 1
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial1;

              #ifdef CONV1_INDEBUG
                if(p==0){

                  cout<<firPartial0<<endl;
                }
              #endif

            }
            if (m_clear & !o_clear) { // 1 0
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial1;

              #ifdef CONV1_INDEBUG
                if(p==0){
                  cout<<firPartial0<<endl;
                }
              #endif

            } 
            if (!m_clear & o_clear) {// 0 1
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];  //debug
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {// 0 0     
              outPartialArr0[p] += firPartial0;   // debug
              outPartialArr1[p] += firPartial1;
            }

            if (o_clear) { //1 // 缓存上一步的结果
              firPartialRes0[p] = firPartial2;  // P2 //这是直接等于  # debug
              firPartialRes1[p] = firPartial3; // P3
            }
            else {
              firPartialRes0[p] += firPartial2; //P2    // 注意这是累加
              firPartialRes1[p] += firPartial3;  // P3
            }

          }	

		if(rep>=SIMDNUM & x_simdnum<CycleNum){
			out_buf0=out_buf0_tmp(MVTU*IM-1,0);
			out_buf1=out_buf1_tmp(MVTU*IM-1,0);
			out.write((out_buf1, out_buf0));
			out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
			out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
		}
		
          if (o_out) {
            for (int p = 0; p < OutP; p++) {
#pragma HLS unroll 
              out_buf0_tmp(p * IM + IM - 1, p * IM) = outPartialArr0[p];
              out_buf1_tmp(p * IM + IM - 1, p * IM) = outPartialArr1[p];
            }
            #ifdef OutMVAU_DEBUG
              cout<<outPartialArr0[9]<<endl;
            #endif
		      }



		if (x_simdnum == SIMDNUM-1) {
			x_simdnum=0;
			if(x_outwnum == OUT_WNUM -1){
				x_outwnum=0;
				if(x_penum == PENUM-1){
					x_penum=0;
				}
				else{
					x_penum++;
				}
			}
			else{
				x_outwnum++;
			}
		}
		else{
			x_simdnum++;
		}

	
	
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out_buf1=out_buf1_tmp(MVTU*IM-1,0);
		out.write((out_buf1, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
		out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
	}


	for (int p = 0; p < OutP; p++) {
	#pragma HLS unroll
		out_buf0_tmp(p * IM + IM - 1, p * IM) = firPartialRes0[p];
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out.write((0, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
	}

}


/*
template <	
			unsigned OUT_H,
			unsigned OUT_W, 
			unsigned K,
			unsigned Cin,
			unsigned Cout,
			unsigned Ibit,
			unsigned Wbit,
			unsigned IM,
			unsigned MVTU,
			unsigned InP,
			unsigned OutP,
			unsigned GUARD_BIT,
			unsigned CASCADE>
void MVAU_DSPOpt_1K(
	stream<ap_uint<InP*Ibit*2> >& vec,
	const ap_uint<InP*K*Wbit> weights[OutP][(Cin*K/InP)*(Cout/OutP)],
	stream<ap_uint<MVTU*IM*2> >& out,
	const unsigned reps = 1){
#pragma HLS ARRAY_PARTITION variable=weights dim=1 complete
	const unsigned PENUM = Cout/OutP;
	const unsigned SIMDNUM = K*Cin/InP;
	const unsigned PROD_BIT = Wbit + Ibit + GUARD_BIT;
	const unsigned WPACK_BIT = Wbit * 3 + Ibit * 2 + GUARD_BIT * 2;
	const unsigned IPACK_BIT = Ibit * 2 + Wbit + GUARD_BIT * 1;
	const unsigned OUT_WNUM = OUT_W / 2;
	const unsigned  CycleNum= OutP/MVTU;




	ap_int<WPACK_BIT> wpacks[OutP][InP];
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wpacks0 dim=2 complete

	ap_uint<IPACK_BIT> ipack[InP];
#pragma HLS ARRAY_PARTITION variable=ipack0 dim=1 complete


	ap_int<IM> firPartialRes0[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes0 dim=1 complete
	ap_int<IM> firPartialRes1[OutP];
#pragma HLS ARRAY_PARTITION variable=firPartialRes1 dim=1 complete

	ap_int<IM> outPartialArr0[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr0 dim=1 complete
	ap_int<IM> outPartialArr1[OutP];
#pragma HLS ARRAY_PARTITION variable=outPartialArr1 dim=1 complete

	ap_int<PROD_BIT + 5> firPartial0;
	ap_int<PROD_BIT + 5> firPartial1;
	ap_int<PROD_BIT + 5> firPartial2;
	ap_int<PROD_BIT + 5> firPartial3;


	unsigned x_simdnum = 0;
	unsigned x_outwnum = 0;
	unsigned x_penum = 0;

	ap_uint<MVTU * IM> out_buf0;
	ap_uint<MVTU * IM> out_buf1;

	ap_uint<OutP * IM> out_buf0_tmp;
	ap_uint<OutP * IM> out_buf1_tmp;

  ap_uint<InP *2* Ibit> data;  // 拿 3*SIMD

	const unsigned long long totalReps = reps*OUT_H*PENUM*OUT_WNUM*SIMDNUM;


	for (unsigned long long rep = 0; rep <totalReps; rep++) {  //50*50
#pragma HLS loop_tripcount min=OUT_H*PENUM*OUT_WNUM*SIMDNUM max=OUT_H*PENUM*OUT_WNUM*SIMDNUM
#pragma HLS PIPELINE II=1

	    bool m_clear = (x_outwnum == 0);
      bool o_clear = (x_simdnum == 0);
      bool o_out = (x_simdnum == SIMDNUM - 1);
	
          
#pragma HLS ARRAY_PARTITION variable=data0 dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data1 dim=1 complete
          data = vec.read();
          pack_input_data_1K<Ibit, InP, PROD_BIT>(data, ipack);


        #ifdef INDEBUG
          cout<<"input act begin-------"<<endl;
          for(int i=0;i< 2*InP;i++){
            ap_uint<4> tmp=data(i * 4 + 4 - 1, i * 4);
            cout<<tmp<<endl;
          }
          cout<<"input act end-------"<<endl;

          cout<<"input pack act begin-------"<<endl;
          for(int i=0;i< InP;i++){
            cout<<ipack[i]<<endl;
          }
          cout<<"input pack act end-------"<<endl;

        #endif

          for (unsigned p = 0; p < OutP; p++) {
#pragma HLS unroll
            pack_weight_data_1K<Wbit, InP, PROD_BIT>(weights[p][x_penum * SIMDNUM + x_simdnum], wpacks[p]);
          }

        #ifdef INDEBUG
        ap_uint<3*Wbit * InP> w;
        ap_int<Wbit> w2_seg ;
        ap_int<Wbit> w1_seg ;
        ap_int<Wbit> w0_seg ;
        cout<<"input w begin-------"<<endl;
        w= weights[0][x_penum * SIMDNUM + x_simdnum];
        for (int i = 0; i < InP; i++) {
            (w2_seg,w1_seg,w0_seg)=w((i+1)*3*Wbit - 1, 3*i * Wbit);
            cout<<w0_seg<<endl;
            cout<<w1_seg<<endl;
            cout<<w2_seg<<endl;
        }
        cout<<"input w end-------"<<endl;
        #endif


          for (int p = 0; p < OutP; p++) {
            // cout << "FIR result compare " << endl;
#pragma HLS unroll 

            simd_MAC<Wbit, Ibit, PROD_BIT, InP, CASCADE>(
                wpacks[p], ipack, firPartial0, firPartial1, firPartial2,
                firPartial3);
            

            // getchar();
            if (m_clear & o_clear) {  // 1 1
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial1;

              #ifdef CONV1_INDEBUG
                if(p==0){

                  cout<<firPartial0<<endl;
                }
              #endif

            }
            if (m_clear & !o_clear) { // 1 0
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial1;

              #ifdef CONV1_INDEBUG
                if(p==0){
                  cout<<firPartial0<<endl;
                }
              #endif

            } 
            if (!m_clear & o_clear) {// 0 1
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];  //debug
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {// 0 0     
              outPartialArr0[p] += firPartial0;   // debug
              outPartialArr1[p] += firPartial1;
            }

            if (o_clear) { //1 // 缓存上一步的结果
              firPartialRes0[p] = firPartial2;  // P2 //这是直接等于  # debug
              firPartialRes1[p] = firPartial3; // P3
            }
            else {
              firPartialRes0[p] += firPartial2; //P2    // 注意这是累加
              firPartialRes1[p] += firPartial3;  // P3
            }

          }	

		if(rep>=SIMDNUM & x_simdnum<CycleNum){
			out_buf0=out_buf0_tmp(MVTU*IM-1,0);
			out_buf1=out_buf1_tmp(MVTU*IM-1,0);
			out.write((out_buf1, out_buf0));
			out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
			out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
		}
		
          if (o_out) {
            for (int p = 0; p < OutP; p++) {
#pragma HLS unroll 
              out_buf0_tmp(p * IM + IM - 1, p * IM) = outPartialArr0[p];
              out_buf1_tmp(p * IM + IM - 1, p * IM) = outPartialArr1[p];
            }
            #ifdef OutMVAU_DEBUG
              cout<<outPartialArr0[9]<<endl;
            #endif
		      }



		if (x_simdnum == SIMDNUM-1) {
			x_simdnum=0;
			if(x_outwnum == OUT_WNUM -1){
				x_outwnum=0;
				if(x_penum == PENUM-1){
					x_penum=0;
				}
				else{
					x_penum++;
				}
			}
			else{
				x_outwnum++;
			}
		}
		else{
			x_simdnum++;
		}

	
	
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out_buf1=out_buf1_tmp(MVTU*IM-1,0);
		out.write((out_buf1, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
		out_buf1_tmp=out_buf1_tmp>>(MVTU*IM);
	}


	for (int p = 0; p < OutP; p++) {
	#pragma HLS unroll
		out_buf0_tmp(p * IM + IM - 1, p * IM) = firPartialRes0[p];
	}

	for (unsigned i = 0; i < CycleNum; i++) {  //2
#pragma HLS PIPELINE II=1
		out_buf0=out_buf0_tmp(MVTU*IM-1,0);
		out.write((0, out_buf0));
		out_buf0_tmp=out_buf0_tmp>>(MVTU*IM);
	}

}
*/

template <unsigned IN_BIT, unsigned W_BIT, unsigned PROD_BIT, unsigned SIMD>
void simd_mac_DSP2(ap_uint<IN_BIT> invec[SIMD], ap_int<W_BIT> w0vec[SIMD],
                   ap_int<W_BIT> w1vec[SIMD], ap_int<PROD_BIT> &out0,
                   ap_int<PROD_BIT> &out1) {
#pragma HLS pipeline
#pragma HLS ARRAY_PARTITION variable=invec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=w0vec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=w1vec dim=1 complete
  ap_int<PROD_BIT * 2> acc = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<PROD_BIT + W_BIT + 1> rst = w1vec[i] * (1 << PROD_BIT) + w0vec[i];
    ap_int<PROD_BIT * 2> m = invec[i] * rst;
    acc += m;
  }

  out0 = acc(PROD_BIT - 1, 0);
  out1 = acc(PROD_BIT * 2 - 1, PROD_BIT) + acc[PROD_BIT - 1];
}


template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned IN_BIT,
          unsigned IN_CH,

          unsigned W_BIT, unsigned B_BIT, unsigned M_BIT,

          unsigned SIMD, unsigned PE>
void MVAU1x1DSP(stream<ap_uint<IN_BIT * SIMD>> &in,
                 const ap_uint<SIMD * W_BIT>
                     weights[PE][((IN_CH * 1) / SIMD) * (OUT_CH / PE)],
                 const ap_int<B_BIT> bias[PE][OUT_CH / PE],
                 stream<ap_uint<PE * M_BIT>> &out, const unsigned reps = 1) {
  const unsigned PROD_BIT = IN_BIT + W_BIT + 2;

#pragma HLS ARRAY_PARTITION variable=bias dim=1 complete

  ap_int<W_BIT> wvec[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable=wvec dim=1 complete
#pragma HLS ARRAY_PARTITION variable=wvec dim=2 complete

  ap_uint<IN_BIT> ivec[SIMD];
#pragma HLS ARRAY_PARTITION variable=ivec dim=1 complete

  ap_int<M_BIT> outPartialArr[PE];
#pragma HLS ARRAY_PARTITION variable=outPartialArr dim=1 complete

  for (unsigned int h = 0; h < OUT_ROW * reps; h++) {
#pragma HLS loop_tripcount min=OUT_ROW max=OUT_ROW
    for (unsigned int w = 0; w < OUT_COL; w++) {
      for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
        for (unsigned int simdIdx = 0; simdIdx < IN_CH / SIMD; simdIdx++) {
#pragma HLS PIPELINE II=1
          ap_uint<IN_BIT *SIMD> inData = in.read();
          for (int s = 0; s < SIMD; s++) {
            ivec[s] = inData((s + 1) * IN_BIT - 1, s * IN_BIT);
          }
          for (int i = 0; i < PE; i++) {
            for (int s = 0; s < SIMD; s++) {
              wvec[i][s] = weights[i][peIdx * IN_CH / SIMD + simdIdx](
                  (s + 1) * W_BIT - 1, s * W_BIT);
            }
          }
          // cout << "w,kc:" << w << "," << kc << endl;

          for (int p = 0; p < PE; p += 2) {
            ap_int<PROD_BIT> outPartial0;
            ap_int<PROD_BIT> outPartial1;
            simd_mac_DSP2<IN_BIT, W_BIT, PROD_BIT, SIMD>(
                ivec, wvec[p], wvec[p + 1], outPartial0, outPartial1);
            if (simdIdx == 0) {
              outPartialArr[p] = outPartial0;
              outPartialArr[p + 1] = outPartial1;
            } else {
              outPartialArr[p] += outPartial0;
              outPartialArr[p + 1] += outPartial1;
            }
          }
          ap_uint<M_BIT * PE> odata;
          if (simdIdx == IN_CH / SIMD - 1) {
            for (int i = 0; i < PE; i++) {
              odata((i + 1) * M_BIT - 1, i * M_BIT) =
                  outPartialArr[i] + bias[i][peIdx];
            }
            out.write(odata);
          }
        }
    }
  }
}
