#pragma once
#include "/tools/Xilinx/Vitis_HLS/2020.2/include/gmp.h"
#define __gmp_const const
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;

#define PAD_DEBUG
// #define pading1_INDEBUG
template <unsigned IN_W, unsigned IN_BIT>
void stream_in_row_l0(stream<ap_uint<3 * IN_BIT>> &in,
                      ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                      bool skip_flag, ap_uint<2> rowBufferIdx) {

  if (skip_flag)
    return;

  for (unsigned w = 0; w < IN_W + 2; w++) {
#pragma HLS pipeline
    ap_uint<3 * IN_BIT> data;
    if (w != 0 && w != IN_W + 1) {
      data = in.read();
    } else {
      data = 0;
    }
    row_buffer[rowBufferIdx][w] = data;
  }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_BIT, unsigned OUTPENUM>
void stream_out_data_l0(stream<ap_uint<3 * IN_BIT * 3 * 3>> &out,
                        ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                        bool skip_flag, ap_int<12> outRowIdx, //outRowIdx=RowIdx
                        ap_uint<2> centerRowBufferIdx) { // centerRowBufferIdx=load
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 factor=3 cyclic

  if (skip_flag)
    return;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) //OUTPENUM=OUT_CH / PE = 16/8=2
    for (unsigned c = 0; c < IN_W; c++){  // IN_W=320
#pragma HLS PIPELINE II=1
        ap_uint<3 * IN_BIT> data[12];
#pragma HLS ARRAY_PARTITION variable=data dim=1 complete
        for (unsigned i = 0; i < 4; i++) {
          data[i] = row_buffer[i][c];
          data[i+4] = row_buffer[i][c + 1];
          data[i+8] = row_buffer[i][c + 2];
        }  // 取一个窗口的数据
        ap_uint<2> row_sel0, row_sel1, row_sel2;
        row_sel0 = centerRowBufferIdx - 1;  // centerRowBufferIdx=0时，-1 也就是3
        row_sel1 = centerRowBufferIdx; // centerRowBufferIdx=0时，就是0
        row_sel2 = centerRowBufferIdx + 1; // centerRowBufferIdx=0时，就是1

    // cout <<"row_sel0: "<<row_sel0<< endl;   
    // cout <<"row_sel1: "<<row_sel1<< endl; 
    // cout <<"row_sel2: "<<row_sel2<< endl;   

        ap_uint<3 * IN_BIT> data00, data01, data02;
        ap_uint<3 * IN_BIT> data10, data11, data12;
        ap_uint<3 * IN_BIT> data20, data21, data22;

        if (outRowIdx - 1 < 0)
        {
          data00 = 0;
          data10 = 0;
          data20 = 0;
        }
        else
        {
          data00 = data[row_sel0];
          data10 = data[row_sel0 + 4];
          data20 = data[row_sel0 + 8];
        }
        data01 = data[row_sel1];
        data11 = data[row_sel1 + 4];
        data21 = data[row_sel1 + 8];  // 行上读三个数
        if (outRowIdx + 1 == IN_H)
        {
          data02 = 0;
          data12 = 0;
          data22 = 0;
        }
        else
        {
          data02 = data[row_sel2];
          data12 = data[row_sel2 + 4];
          data22 = data[row_sel2 + 8];
        }
        // out.write((data22, data21, data20, data12, data11, data10, data02, data01, data00));  // 取窗口3*3的数据，每个数据 3*8bit
        out.write((data22, data12, data02, data21, data11, data01, data20, data10, data00));  // 取窗口3*3的数据，每个数据 3*8bit

    }
}



template <unsigned IN_H, unsigned IN_W, unsigned IN_BIT, unsigned OUTPENUM>
void stream_out_data_l0_opt(stream<ap_uint<3 * IN_BIT * 3 * 3>> &out,
                        ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                        bool skip_flag, ap_int<12> outRowIdx, //outRowIdx=RowIdx
                        ap_uint<2> centerRowBufferIdx) { // centerRowBufferIdx=load
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 factor=3 cyclic

  if (skip_flag)
    return;

    ap_uint<2> row_sel0=centerRowBufferIdx - 1;
    ap_uint<2> row_sel1 = centerRowBufferIdx;
    ap_uint<2> row_sel2 = centerRowBufferIdx + 1;

    ap_uint<3 * IN_BIT> data00, data01, data02;
    ap_uint<3 * IN_BIT> data10, data11, data12;
    ap_uint<3 * IN_BIT> data20, data21, data22;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) //OUTPENUM=OUT_CH / PE = 16/8=2
    for (unsigned c = 0; c < IN_W; c++){  // IN_W=320
#pragma HLS PIPELINE II=1

        if (outRowIdx - 1 < 0)
        {
          data00 = 0;
          data10 = 0;
          data20 = 0;
        }
        else
        {
          data00 = row_buffer[row_sel0][c];
          data10 = row_buffer[row_sel0][c+1];
          data20 = row_buffer[row_sel0][c+2];
        }
        data01 = row_buffer[row_sel1][c];
        data11 = row_buffer[row_sel1][c+1];
        data21 = row_buffer[row_sel1][c+2];  // 行上读三个数
        if (outRowIdx + 1 == IN_H)
        {
          data02 = 0;
          data12 = 0;
          data22 = 0;
        }
        else
        {
          data02 = row_buffer[row_sel2][c];
          data12 = row_buffer[row_sel2][c+1];
          data22 = row_buffer[row_sel2][c+2];
        }
        // out.write((data22, data21, data20, data12, data11, data10, data02, data01, data00));  // 取窗口3*3的数据，每个数据 3*8bit
        out.write((data22, data12, data02, data21, data11, data01, data20, data10, data00));  // 取窗口3*3的数据，每个数据 3*8bit

    }
}



template <unsigned IN_H, unsigned IN_W, unsigned IN_BIT, unsigned OUTPENUM>
void stream_out_data_l0_1K(stream<ap_uint<3 * IN_BIT * 3> > &out,
                        ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 2],
                        bool skip_flag, ap_int<12> outRowIdx, //outRowIdx=RowIdx
                        ap_uint<2> centerRowBufferIdx) { // centerRowBufferIdx=load
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 factor=3 cyclic

  if (skip_flag)
    return;

    ap_uint<2> row_sel0=centerRowBufferIdx - 1;
    // ap_uint<2> row_sel1 = centerRowBufferIdx;
    ap_uint<2> row_sel2 = centerRowBufferIdx + 1;

    ap_uint<3 * IN_BIT> data0, data1, data2;
    ap_uint<IN_BIT> data22,data21, data20, data12,data11,data10,data02,data01,data00;
    unsigned wMat0 = 0;
    unsigned wMat1 = 0;
    const unsigned totalReps = OUTPENUM*IN_W*3;


  for (unsigned peIdx = 0; peIdx < totalReps; peIdx++) {//OUTPENUM=OUT_CH / PE = 16/8=2

#pragma HLS PIPELINE II=1


        if (((outRowIdx - 1 < 0)&& (row_sel0==ap_uint<2>(centerRowBufferIdx - 1)))||((outRowIdx + 1 == IN_H)&& (row_sel0==ap_uint<2>(centerRowBufferIdx + 1))))
        {
          data0 = 0;
          data1 = 0;
          data2 = 0;
        }
        else
        {
          data0 = row_buffer[row_sel0][wMat0];
          data1 = row_buffer[row_sel0][wMat0+1];
          data2 = row_buffer[row_sel0][wMat0+2];
        }


        data00=data0(IN_BIT-1,0);
        data01=data1(IN_BIT-1,0);
        data02=data2(IN_BIT-1,0);

        data10=data0(2*IN_BIT-1,IN_BIT);
        data11=data1(2*IN_BIT-1,IN_BIT);
        data12=data2(2*IN_BIT-1,IN_BIT);

        data20=data0(3*IN_BIT-1,IN_BIT*2);
        data21=data1(3*IN_BIT-1,IN_BIT*2);
        data22=data2(3*IN_BIT-1,IN_BIT*2);

        out.write((data22,data21, data20, data12,data11,data10,data02,data01,data00));  // 取窗口3*3的数据，每个数据 3*8bit


        if(row_sel0==row_sel2){
          row_sel0=centerRowBufferIdx - 1;
          if(wMat0==IN_W-1){
            wMat0=0;
            if(wMat1 == OUTPENUM-1)
              wMat1=0;
            else
              wMat1++;            
          }
          else{
            wMat0++;
          }
        }
        else{
          row_sel0=row_sel0+1;
        }
  }
}



template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned OUTPENUM>
void conv3padding_l0(stream<ap_uint<3 * IN_BIT> > &in,
                     stream<ap_uint<3 * IN_BIT * 3 * 3> > &out,
                     const unsigned reps = 1) {

  ap_uint<IN_CH * IN_BIT> row_buffer[4][IN_W + 2];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p
  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = -2; 
  ap_int<10> rowIdx = -2;

  for (unsigned rep = 0; rep < reps * IN_H + 2; rep++) {
#pragma HLS DEPENDENCE array false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H + 2 max=IN_H + 2
    // std::cout <<"loadBufferIdx:"<<loadBufferIdx<< endl;
    // std::cout <<"storeBufferIdx:"<<storeBufferIdx<< endl; 
    // std::cout <<"rowIdx: " << rowIdx<< std::endl;   

    stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, (rep >= reps * IN_H),
                                   storeBufferIdx);  // 如果大于等于 reps * IN_H 该函数不操作直接返回， 否则读取一行数据（补0后）到row_buffer  // 注意该函数只给行上补0
    stream_out_data_l0_opt<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, (rep < 2), // 如果rep<2,(也就是row_buffer存储满3行)，该函数不操作直接返回，否则    // 注意列补零合并在该函数里
                                                 rowIdx, loadBufferIdx);

    // cout << "out.size(): " << out.size() << endl;
    loadBufferIdx++;  // 虽然一直做加法，但是由于位宽限制，取值重复 2，3，0，1 
    storeBufferIdx++;  // 虽然一直做加法，但是由于位宽限制，取值重复0，1，2，3                                                                                                
    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned OUTPENUM>
void conv3padding_l0_1K(stream<ap_uint<3 * IN_BIT> > &in,
                     stream<ap_uint<3 * IN_BIT * 3> > &out,
                     const unsigned reps = 1) {

  ap_uint<IN_CH * IN_BIT> row_buffer[4][IN_W + 2];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p
  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = -2; 
  ap_int<10> rowIdx = -2;

  for (unsigned rep = 0; rep < reps * IN_H + 2; rep++) {
#pragma HLS DEPENDENCE array false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H + 2 max=IN_H + 2
    // std::cout <<"loadBufferIdx:"<<loadBufferIdx<< endl;
    // std::cout <<"storeBufferIdx:"<<storeBufferIdx<< endl; 
    // std::cout <<"rowIdx: " << rowIdx<< std::endl;   

    stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, (rep >= reps * IN_H),
                                   storeBufferIdx);  // 如果大于等于 reps * IN_H 该函数不操作直接返回， 否则读取一行数据（补0后）到row_buffer  // 注意该函数只给行上补0
    stream_out_data_l0_1K<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, (rep < 2), // 如果rep<2,(也就是row_buffer存储满3行)，该函数不操作直接返回，否则    // 注意列补零合并在该函数里
                                                 rowIdx, loadBufferIdx);

    // cout << "out.size(): " << out.size() << endl;
    loadBufferIdx++;  // 虽然一直做加法，但是由于位宽限制，取值重复 2，3，0，1 
    storeBufferIdx++;  // 虽然一直做加法，但是由于位宽限制，取值重复0，1，2，3                                                                                                
    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
}


template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row(
    stream<ap_uint<IN_PE * IN_BIT * 2> > &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<2> rowBufferIdx) {
#pragma HLS inline off
  if (skip_flag)
    return;
  // ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)  //16/2 通道上并行度
    for (unsigned w = 0; w < IN_W / 2; w++) { // 160/2  行上的并行度
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT * 2> data;
      data = in.read();
      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }

  // #ifdef pading1_INDEBUG
  // cout<<"debug begin"<<endl;
  // ap_uint<IN_PE * IN_BIT * 2> data_test=row_buffer[0][0][1];
  // cout <<"The Value of data: \t" <<data_test<< " \t Binary format: \t" <<data_test.to_string(2).c_str()<< '\n';
  // for(int x=0;x<16;x++){
  //   cout<<data_test((x + 1) * 4 - 1, x * 4)<<endl;
  // }
  // #endif

}

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row_opt(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<2> rowBufferIdx) {
#pragma HLS inline off
  if (skip_flag)
    return;
  const unsigned OUT_WNUM = IN_W / 2;
  const unsigned SIMDNUM = IN_CH/IN_PE;
  unsigned x_simdnum = 0;
  unsigned x_outwnum = 0;
  ap_uint<IN_PE * IN_BIT * 2> data;

  for (unsigned peIdx = 0; peIdx < OUT_WNUM*SIMDNUM; peIdx++) { //16/2 通道上并行度
#pragma HLS PIPELINE II=1
      
      data = in.read();
      row_buffer[x_simdnum % (SIMD / IN_PE)][rowBufferIdx]
                [x_outwnum * IN_CH / SIMD + x_simdnum / (SIMD / IN_PE)] = data;

		if (x_outwnum == OUT_WNUM-1) {
			x_outwnum=0;
			if(x_simdnum == SIMDNUM -1){
				x_simdnum=0;
			}
			else{
				x_simdnum++;
			}
		}
		else{
			x_outwnum++;
		}
  }

}


template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data(
    stream<ap_uint<SIMD * IN_BIT * 2 * K>> &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {  // startRowBufferIdx=startRowBufferIdx
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete


  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;
  if (skip_flag)
    return;

  // ap_uint<8> infoldIdx = 0;
  ap_uint<5> simdIdx = 0;
  ap_uint<8> w = 0;

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) { //Cout/PE
    for (unsigned cycle = 0; cycle < WLEN * SIMDNUM; cycle++) {  // 80/2 * 32/16

#pragma HLS PIPELINE II=1
      ap_uint<SIMD * IN_BIT> data0[K];
#pragma HLS ARRAY_PARTITION variable=data0 dim=1 complete
      ap_uint<SIMD * IN_BIT> data1[K];
#pragma HLS ARRAY_PARTITION variable=data1 dim=1 complete
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[K][SIMD / IN_PE];
#pragma HLS ARRAY_PARTITION variable=buffer_data dim=1 complete
      for (unsigned wr = 0; wr < K; wr++) {
#pragma HLS unroll
        ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
        for (unsigned i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
          buffer_data[wr][i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];  // w++ -> simdIdx == SIMDNUM - 1
        }

        if (outRowIdx - K / 2 + wr < 0 || outRowIdx - K / 2 + wr >= IN_H) {
          data0[wr] = 0;
          data1[wr] = 0;
        } else {
          for (unsigned i = 0; i < SIMD / IN_PE; i++) {
            data0[wr]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
                buffer_data[wr][i](IN_PE_BIT - 1, 0);
            data1[wr]((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
                buffer_data[wr][i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
          }
        }
      }

      out.write((data1[0], data0[0], data1[1], data0[1], data1[2], data0[2]));

      if (cycle == WLEN * SIMDNUM - 1) {
        w = 0;
      } else if (simdIdx == SIMDNUM - 1) {
        w++;
      }

      if (simdIdx == SIMDNUM - 1) {
        simdIdx = 0;
      } else {
        simdIdx++;
      }
    }
  }
}



template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data_1K(
    stream<ap_uint<SIMD * IN_BIT * 2> > &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {  // startRowBufferIdx=startRowBufferIdx
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete


  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;
  if (skip_flag)
    return;

  // ap_uint<8> infoldIdx = 0;
    unsigned simdIdx = 0;
    unsigned w = 0;
    unsigned wr = 0;
    ap_uint<2 * SIMD * IN_BIT> data;

    ap_uint<2> rowBufferIdx = startRowBufferIdx;
  for (unsigned peIdx = 0; peIdx < OUTPENUM*WLEN * K * SIMDNUM; peIdx++) { //Cout/PE
  // 80/2 * 32/16

#pragma HLS PIPELINE II=1
        if (outRowIdx - K / 2 + wr < 0 || outRowIdx - K / 2 + wr >= IN_H) {
          data = 0;
        }
        else{
          rowBufferIdx=startRowBufferIdx+wr;
          for (unsigned i = 0; i < SIMD / IN_PE; i++) {
  #pragma HLS unroll
            data((i + 1) * 2* IN_PE_BIT - 1, i * 2* IN_PE_BIT) = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];  // w++ -> simdIdx == SIMDNUM - 1
          }
        }

      out.write(data);


      #ifdef pading1_INDEBUG
          if(peIdx>=SIMDNUM){
            cout<<"debug begin"<<endl;
            cout <<"The Value of data: \t" <<data_test<<endl;
            ap_uint<2 * SIMD * IN_BIT> data_test=row_buffer[0][rowBufferIdx][w * SIMDNUM + simdIdx];
            cout <<"The Value of data: \t" <<data_test<< " \t Binary format: \t" <<data_test.to_string(2).c_str()<< '\n';
            cout <<"The Value of data: \t" <<data<< " \t Binary format: \t" <<data.to_string(2).c_str()<< '\n';
            for(int x=0;x<16;x++){
              cout<<data((x + 1) * 4 - 1, x * 4)<<endl;
            }
          }
      #endif




      if(simdIdx == SIMDNUM - 1){
        simdIdx=0;
        if(wr==K-1){
          wr=0;
          if(w==WLEN-1){
            w=0;
          }
          else{
            w++;
          }
        }
        else{
          wr++;
        }
      }
      else{
        simdIdx++;
      }

  }
}



template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                  stream<ap_uint<SIMD * IN_BIT * 2 * K>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0"); // 本层输入并行度大于等于下一层的并行度
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [IN_W / 2 * IN_CH / SIMD];
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

//#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS DEPENDENCE false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H - 2 max=IN_H - 2
    // std::cout <<"loadBufferIdx:"<<loadBufferIdx<< endl;
    // std::cout <<"storeBufferIdx:"<<storeBufferIdx<< endl; 
    // std::cout <<"rowIdx: " << rowIdx<< std::endl;   

    stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, false, storeBufferIdx);
    stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, false, rowIdx, loadBufferIdx);

    
    loadBufferIdx++;

    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}


template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding_1K(stream<ap_uint<IN_PE * IN_BIT * 2> > &in,
                  stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0"); // 本层输入并行度大于等于下一层的并行度
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [IN_W / 2 * IN_CH / SIMD];
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=2 complete

//#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  stream_in_row_opt<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row_opt<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, false, storeBufferIdx);
  storeBufferIdx++;
  
  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS DEPENDENCE false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H - 2 max=IN_H - 2
    // std::cout <<"loadBufferIdx:"<<loadBufferIdx<< endl;
    // std::cout <<"storeBufferIdx:"<<storeBufferIdx<< endl; 
    // std::cout <<"rowIdx: " << rowIdx<< std::endl;   

    stream_in_row_opt<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, false, storeBufferIdx);
    stream_out_data_1K<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, false, rowIdx, loadBufferIdx);

    
    loadBufferIdx++;

    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data_1K<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data_1K<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, false, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}



template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void streamInOneRowTwoPix(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][2][2]
                                      [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<1> rowBufferIdx) {
#pragma HLS inline off
  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;

  if (skip_flag)
    return;
  static ap_uint<IN_PE *IN_BIT> reg = 0;

  for (unsigned s = 0; s < SIMDNUM; s++)
    for (unsigned p = 0; p < INPENUM; p++)
      for (unsigned w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline
        ap_uint<IN_PE * IN_BIT> data1, data0;
        (data1, data0) = in.read();

        row_buffer[p][0][rowBufferIdx][w * SIMDNUM + s] = data0;
        row_buffer[p][1][rowBufferIdx][w * SIMDNUM + s] = data1;
      }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void streamOutOneRowTwoPix(
    stream<ap_uint<SIMD * IN_BIT>> &out,
    ap_uint<IN_PE * IN_BIT> row_buffer[SIMD / IN_PE][2][2]
                                      [IN_W / 2 * IN_CH / SIMD],
    bool skip_flag, ap_uint<1> rowBufferIdx) {
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
  const unsigned IN_PE_BIT = IN_PE * IN_BIT;

  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  if (skip_flag)
    return;

  for (unsigned w = 0; w < IN_W; w++) {
    for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++) {
      for (unsigned s = 0; s < SIMDNUM; s++) {
#pragma HLS pipeline
        ap_uint<SIMD * IN_BIT> data;
        ap_uint<IN_PE * IN_BIT> buffer_data[INPENUM];
#pragma HLS ARRAY_PARTITION variable=buffer_data dim=1 complete
        ap_uint<1> sel = w % 2;

        for (unsigned i = 0; i < INPENUM; i++) {
          buffer_data[i] =
              row_buffer[i][sel][rowBufferIdx][w / 2 * SIMDNUM + s];
        }

        for (unsigned p = 0; p < INPENUM; p++) {
          data((p + 1) * IN_PE_BIT - 1, p * IN_PE_BIT) = buffer_data[p];
        }
        out.write(data);
      }
    }
  }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_CH, unsigned IN_BIT,
          unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv1x1convert(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                    stream<ap_uint<SIMD * IN_BIT>> &out,
                    const unsigned reps = 1) {

  const unsigned INPENUM = SIMD / IN_PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  ap_uint<IN_PE * IN_BIT> row_buffer[INPENUM][2][2][IN_W / 2 * SIMDNUM];
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p

#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete
// #pragma HLS ARRAY_PARTITION variable = row_buffer dim = 2 complete
//#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<1> storeBufferIdx = 0;
  ap_uint<1> loadBufferIdx = 1;

  for (unsigned rep = 0; rep < reps * IN_H + 1; rep++) {
#pragma HLS DEPENDENCE false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H + 1 max=IN_H + 1
    streamInOneRowTwoPix<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, (rep >= reps * IN_H), storeBufferIdx);
    streamOutOneRowTwoPix<IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, (rep == 0), loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;
  }
}
