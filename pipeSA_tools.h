#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;
//#define IN_DEBUG
#define PRINT_8_DEBUG

#define MAX(x, y) (((x) > (y)) ? (x) : (y)) /* \brief Maximum value between x and y*/
#define MAX_W 160  // 实际为W/2
#define MAX_FACTOR 5
#define MAX_BUF_LENGTH 512



template <unsigned IN_BIT, unsigned PE>
ap_uint<IN_BIT * PE> max2_PE(ap_uint<IN_BIT * PE> data0,
                             ap_uint<IN_BIT * PE> data1) {
  ap_uint<IN_BIT * PE> ret;

  for (int i = 0; i < PE; i++) {
    ap_uint<IN_BIT> d0 = data0(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> d1 = data1(IN_BIT * (i + 1) - 1, IN_BIT * i);
    ap_uint<IN_BIT> dret = d1 > d0 ? d1 : d0;
    ret(IN_BIT * (i + 1) - 1, IN_BIT * i) = dret;
  }
  return ret;
}

template <unsigned IN_BIT, unsigned PE>
ap_uint<IN_BIT * PE> max2_PE_1K_ROW(ap_uint<IN_BIT * PE *2 > data_in) {
  ap_uint<IN_BIT * PE> ret;

  for (int i = 0; i < PE; i++) {
	ap_uint<IN_BIT> d0 = data_in(IN_BIT * (2*i + 1) - 1, IN_BIT * i *2);
	ap_uint<IN_BIT> d1 = data_in(IN_BIT * (2*i + 2) - 1, IN_BIT * (2*i+1));
    ap_uint<IN_BIT> dret = d1 > d0 ? d1 : d0;
    ret(IN_BIT * (i + 1) - 1, IN_BIT * i) = dret;
  }
  return ret;
}

template <unsigned PaddingUp,
      unsigned PaddingDown,
      unsigned SIMD,
			unsigned IN_BIT
>
void SAMEPAD_DSPopt_SA_UP_DOWN(
	stream<ap_uint<SIMD * IN_BIT * 2> >& in,
	stream<ap_uint<SIMD * IN_BIT * 2> >& out,
  const unsigned Din_H,
  const unsigned Din_W_TRUE,
  const unsigned Cin){
    ap_uint<SIMD * IN_BIT * 2> outData;
    ap_uint<SIMD * IN_BIT * 2> inData;

  for(unsigned int y = 0; y<Din_H; y++){
    for(unsigned int k = 0; k<Cin/SIMD; k++){
      for(unsigned int x=0; x < Din_W_TRUE; x++){
#pragma HLS PIPELINE II=1  
        // padding rows
        if(y< PaddingUp||y>=Din_H-PaddingDown){
          outData = 0;
        }
        else{
          outData=in.read();
        }
        out.write(outData);
      }
    }
  }
}


template <unsigned PaddingUp,
      unsigned SIMD,
			unsigned IN_BIT
>
void SAMEPAD_DSPopt_SA_UP(
	stream<ap_uint<SIMD * IN_BIT * 2> >& in,
	stream<ap_uint<SIMD * IN_BIT * 2> >& out,
  bool skip_flag,
  const unsigned Din_H,
  const unsigned Din_W_TRUE,
  const unsigned Cin){
    ap_uint<SIMD * IN_BIT * 2> outData;
    ap_uint<SIMD * IN_BIT * 2> inData;

  if (skip_flag)
    return;
  for(unsigned int y = 0; y<Din_H; y++){
    for(unsigned int k = 0; k<Cin/SIMD; k++){
      for(unsigned int x=0; x < Din_W_TRUE; x++){
#pragma HLS PIPELINE II=1  
        // padding rows
        if(y< PaddingUp){
          outData = 0;
        }
        else{
          outData=in.read();
        }
        out.write(outData);
      }
    }
  }
}



template <unsigned PaddingDown,
      unsigned SIMD,
			unsigned IN_BIT
>
void SAMEPAD_DSPopt_SA_DOWN(
	stream<ap_uint<SIMD * IN_BIT * 2> >& in,
	stream<ap_uint<SIMD * IN_BIT * 2> >& out,
  bool skip_flag,
  const unsigned Din_H,
  const unsigned Din_W_TRUE,
  const unsigned Cin){
  if (skip_flag)
    return;
    ap_uint<SIMD * IN_BIT * 2> outData;
    ap_uint<SIMD * IN_BIT * 2> inData;


    for(unsigned int y = 0; y<Din_H; y++){
      for(unsigned int k = 0; k<Cin/SIMD; k++){
        for(unsigned int x=0; x < Din_W_TRUE; x++){
  #pragma HLS PIPELINE II=1  
          // padding rows
          if(y>=Din_H-PaddingDown){
            outData = 0;
          }
          else{
            outData=in.read();
          }
          out.write(outData);
        }
      }
    }

}

/*
template <unsigned K,unsigned IN_BIT, unsigned SIMD> // 注意这里的IN_H是padding后的
void conv3padding_opt_SA(stream<ap_uint<SIMD * IN_BIT * 2> > &in,
                     stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                     const unsigned IN_H,
                     const unsigned IN_W,
                     const unsigned OUT_H,
                     const unsigned IN_CH,
                     const unsigned OUTPENUM) {

  const unsigned int multiplying_factor = IN_CH/SIMD;
  const unsigned int number_blocks = K + 1 ;

  ap_uint<SIMD * IN_BIT * 2> row_buffer[4][(MAX_W/2)*MAX_FACTOR];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p

  const unsigned int cycles_write_block = OUTPENUM * (IN_W/2) * K *multiplying_factor; // 一次读一行的三个
  const unsigned int cycles_read_block = (IN_W/2)*multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  const unsigned int baseIter = (IN_W/2) * K *multiplying_factor // Initial buffer
			                  + OUT_H * MAX(cycles_write_block,cycles_read_block);

  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, wMat =0,count_simd=0;
  unsigned int counter_internal_block = 0;

  unsigned int current_block_write = 0;
  unsigned int current_block_read = 0;
  unsigned int block_read_K;
  unsigned int current_line = 0;
  unsigned int current_line_w = 0;
  unsigned int current_line_simd = 0;
  unsigned int read_block = 0; 
  unsigned int current_line_in_block;
  unsigned int  flag = 0; 
  ap_uint<SIMD * IN_BIT * 2> inElem;
  ap_uint<2 * SIMD * IN_BIT> data;
  #ifdef INPAD_DEBUG
    unsigned int m=0;
  #endif

  for (unsigned rep = 0; rep < baseIter; rep++) {
#pragma HLS PIPELINE II=1   
    if (inp < K* (IN_W/2)*multiplying_factor) {// Initial buffer of ConvKernelDim lines	
        inElem = in.read();

        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;
        inp++;

        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              current_block_write++;
            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
    }
    else{
      if(counter_internal_block < cycles_write_block){
        block_read_K=current_block_read+k_y;
        if (block_read_K >= number_blocks) {
          block_read_K-= number_blocks;
        }

        current_line_in_block = ofm_x*multiplying_factor+count_simd;

        data=row_buffer[block_read_K][(current_line_in_block)];


        out.write(data);
        #ifdef INPAD_DEBUG
            if(m==10768){
              cout<<"debug...."<<endl;
              cout<<data<<endl;
            }
            m++;
        #endif

        if(ofm_x==IN_W/2-1){
          ofm_x=0;
          if(count_simd==multiplying_factor-1){
            count_simd=0;
            if(k_y==K-1){
              k_y=0;
              if(wMat==OUTPENUM-1){
                wMat=0;
                current_block_read++;
                if(current_block_read>=number_blocks){
                  current_block_read-= number_blocks;
                }
              }
              else{
                wMat++;
              }              
            }
            else{
              k_y++;
            }
          }
          else{
            count_simd++;
          }
        }
        else{
          ofm_x++;
        }    
      }
      if ((counter_internal_block < cycles_read_block) && (read_block<IN_H)) {
        inElem=in.read();
        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;


        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              if (current_block_write == number_blocks-1) {
                current_block_write=0;
              }
              else{
                current_block_write++;
              }
            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
      }


      if(counter_internal_block == (max_cycles-1)){
        counter_internal_block = 0;
      }
      else{
        counter_internal_block++; 
      }

    }
    if(flag==baseIter-1){
          flag = 0; 
          inp=0;
          read_block=0;
    }
    else{
      flag++; 
    }

  }
}

*/


template <unsigned K,unsigned IN_BIT, unsigned SIMD> // 注意这里的IN_H是padding后的
void conv3padding_opt_SA(stream<ap_uint<SIMD * IN_BIT * 2> > &in,
                     stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                     const unsigned IN_H,
                     const unsigned IN_W,
                     const unsigned OUT_H,
                     const unsigned IN_CH,
                     const unsigned OUTPENUM) {

  const unsigned int multiplying_factor = IN_CH/SIMD;
  const unsigned int number_blocks = K + 1 ;

  ap_uint<SIMD * IN_BIT * 2> row_buffer[4][MAX_BUF_LENGTH];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p

  const unsigned int cycles_write_block = OUTPENUM * (IN_W/2) * K *multiplying_factor; // 一次读一行的三个
  const unsigned int cycles_read_block = (IN_W/2)*multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  const unsigned int baseIter = (IN_W/2) * K *multiplying_factor // Initial buffer
			                  + OUT_H * MAX(cycles_write_block,cycles_read_block);

  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, wMat =0,count_simd=0;
  unsigned int counter_internal_block = 0;

  ap_uint<2> current_block_write = 0;
  ap_uint<2> current_block_read = 0;

  ap_uint<2> block_read_K;

  unsigned int current_line = 0;
  unsigned int current_line_w = 0;
  unsigned int current_line_simd = 0;
  unsigned int read_block = 0; 
  unsigned int current_line_in_block;
  unsigned int  flag = 0; 
  ap_uint<SIMD * IN_BIT * 2> inElem;
  ap_uint<2 * SIMD * IN_BIT> data;
  #ifdef INPAD_DEBUG
    unsigned int m=0;
  #endif

  for (unsigned rep = 0; rep < baseIter; rep++) {
#pragma HLS PIPELINE II=1   
    if (inp < K* (IN_W/2)*multiplying_factor) {// Initial buffer of ConvKernelDim lines	
        inElem = in.read();

        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;
        inp++;

        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              current_block_write++;
            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
    }
    else{
      if(counter_internal_block < cycles_write_block){
        block_read_K=current_block_read+k_y;


        current_line_in_block = ofm_x*multiplying_factor+count_simd;

        data=row_buffer[block_read_K][(current_line_in_block)];


        out.write(data);
        #ifdef INPAD_DEBUG
            if(m==10768){
              cout<<"debug...."<<endl;
              cout<<data<<endl;
            }
            m++;
        #endif

        if(ofm_x==IN_W/2-1){
          ofm_x=0;
          if(count_simd==multiplying_factor-1){
            count_simd=0;
            if(k_y==K-1){
              k_y=0;
              if(wMat==OUTPENUM-1){
                wMat=0;
                current_block_read++;
              }
              else{
                wMat++;
              }              
            }
            else{
              k_y++;
            }
          }
          else{
            count_simd++;
          }
        }
        else{
          ofm_x++;
        }    
      }
      if ((counter_internal_block < cycles_read_block) && (read_block<IN_H)) {
        inElem=in.read();
        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;


        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              current_block_write++;

            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
      }


      if(counter_internal_block == (max_cycles-1)){
        counter_internal_block = 0;
      }
      else{
        counter_internal_block++; 
      }

    }
    if(flag==baseIter-1){
          flag = 0; 
          inp=0;
          read_block=0;
    }
    else{
      flag++; 
    }

  }
}



/**
 * version: 20230617 该版本是流水用的，取出数据的顺序部队
*/
template <unsigned K,unsigned IN_BIT, unsigned SIMD> // 注意这里的IN_H是padding后的
void conv3padding_opt_SA_V0(stream<ap_uint<SIMD * IN_BIT * 2> > &in,
                     stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                     const unsigned IN_H,
                     const unsigned IN_W,
                     const unsigned OUT_H,
                     const unsigned IN_CH,
                     const unsigned OUTPENUM,
                     const unsigned reps = 1) {

  const unsigned int multiplying_factor = IN_CH/SIMD;
  const unsigned int number_blocks = K + 1 ;

  ap_uint<SIMD * IN_BIT * 2> row_buffer[4][(MAX_BUF_LENGTH/2)*multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p

  const unsigned int cycles_write_block = OUTPENUM * (IN_W/2) * K *multiplying_factor; // 一次读一行的三个
  const unsigned int cycles_read_block = (IN_W/2)*multiplying_factor;
  const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
  const unsigned int baseIter = (IN_W/2) * K *multiplying_factor // Initial buffer
			                  + OUT_H * MAX(cycles_write_block,cycles_read_block);

  unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, wMat =0,count_simd=0;
  unsigned int counter_internal_block = 0;

  unsigned int current_block_write = 0;
  unsigned int current_block_read = 0;
  unsigned int block_read_K;
  unsigned int current_line = 0;
  unsigned int current_line_w = 0;
  unsigned int current_line_simd = 0;
  unsigned int read_block = 0; 
  unsigned int current_line_in_block;
  unsigned int  flag = 0; 
  ap_uint<SIMD * IN_BIT * 2> inElem;
  ap_uint<2 * SIMD * IN_BIT> data;
  #ifdef INPAD_DEBUG
    unsigned int m=0;
  #endif

  for (unsigned rep = 0; rep < reps*baseIter; rep++) {
#pragma HLS PIPELINE II=1   
    if (inp < K* (IN_W/2)*multiplying_factor) {// Initial buffer of ConvKernelDim lines	
        inElem = in.read();

        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;
        inp++;

        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              current_block_write++;
            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
    }
    else{
      if(counter_internal_block < cycles_write_block){
        block_read_K=current_block_read+k_y;
        if (block_read_K >= number_blocks) {
          block_read_K-= number_blocks;
        }

        current_line_in_block = ofm_x*multiplying_factor+count_simd;

        data=row_buffer[block_read_K][(current_line_in_block)];


        out.write(data);
        #ifdef INPAD_DEBUG
            if(m==10768){
              cout<<"debug...."<<endl;
              cout<<data<<endl;
            }
            m++;
        #endif
        if(count_simd==multiplying_factor-1){
          count_simd=0;
          if(k_y==K-1){
            k_y=0;
            if(ofm_x==IN_W/2-1){
              ofm_x=0;
              if(wMat==OUTPENUM-1){
                wMat=0;
                current_block_read++;
                if(current_block_read>=number_blocks){
                  current_block_read-= number_blocks;
                }
              }
              else{
                wMat++;
              }
            }
            else{
              ofm_x++;
            }
          }
          else{
            k_y++;
          }
        }
        else{
          count_simd++;
        }      
      }
      if ((counter_internal_block < cycles_read_block) && (read_block<IN_H)) {
        inElem=in.read();
        row_buffer[current_block_write][current_line_w * (IN_CH / SIMD) + current_line_simd] = inElem;


        if(current_line_w==IN_W/2-1){
            current_line_w=0;
            if(current_line_simd==multiplying_factor-1){
              current_line_simd=0;
              read_block++;
              if (current_block_write == number_blocks-1) {
                current_block_write=0;
              }
              else{
                current_block_write++;
              }
            }
            else{
              current_line_simd++;
            }
        }
        else{
          current_line_w++;
        }
      }


      if(counter_internal_block == (max_cycles-1)){
        counter_internal_block = 0;
      }
      else{
        counter_internal_block++; 
      }

    }
    if(flag==baseIter-1){
          flag = 0; 
          inp=0;
          read_block=0;
    }
    else{
      flag++; 
    }

  }
}







template <unsigned IN_BIT, unsigned SIMD>
void stream_in_row_SA(
    stream<ap_uint<SIMD * IN_BIT * 2>> &in,
    ap_uint<SIMD * IN_BIT * 2> row_buffer[4][MAX_BUF_LENGTH],
    bool skip_flag, ap_uint<2> rowBufferIdx,
    const unsigned IN_W,
    const unsigned IN_CH) {
#pragma HLS inline off
  if (skip_flag)
    return;
  const unsigned OUT_WNUM = IN_W / 2;
  const unsigned SIMDNUM = IN_CH/SIMD;
  unsigned x_simdnum = 0;
  unsigned x_outwnum = 0;
  ap_uint<SIMD * IN_BIT * 2> data;

  for (unsigned peIdx = 0; peIdx < OUT_WNUM*SIMDNUM; peIdx++) { //16/2 通道上并行度
#pragma HLS PIPELINE II=1
      data = in.read();
      row_buffer[rowBufferIdx][x_outwnum * SIMDNUM+ x_simdnum] = data;

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


template <unsigned K, unsigned IN_BIT, unsigned SIMD>
void stream_out_data_SA(
    stream<ap_uint<SIMD * IN_BIT * 2> > &out,
    ap_uint<SIMD * IN_BIT * 2> row_buffer[4][MAX_BUF_LENGTH],
    bool skip_flag, ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx,
    const unsigned IN_H,
    const unsigned IN_W,
    const unsigned IN_CH,
    const unsigned OUTPENUM ) {  // startRowBufferIdx=startRowBufferIdx
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete


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

          data = row_buffer[rowBufferIdx][w * SIMDNUM + simdIdx];  // w++ -> simdIdx == SIMDNUM - 1

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

template <unsigned K, unsigned IN_BIT, unsigned SIMD>
void conv3padding_SA(stream<ap_uint<SIMD * IN_BIT * 2> > &in,
                  stream<ap_uint<SIMD * IN_BIT * 2> > &out,
                  const unsigned IN_H,
                  const unsigned IN_W,
                  const unsigned IN_CH,
                  const unsigned OUTPENUM,
                  const unsigned reps = 1) {

  static_assert(K == 3, "K!=3");

  ap_uint<SIMD * IN_BIT * 2> row_buffer[4][MAX_BUF_LENGTH];
#pragma HLS BIND_STORAGE variable=row_buffer type=ram_s2p
#pragma HLS ARRAY_PARTITION variable=row_buffer dim=1 complete

//#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM
  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;

  stream_in_row_SA<IN_BIT, SIMD>(
    in, row_buffer, false, storeBufferIdx,IN_W,IN_CH);
  storeBufferIdx++;
  
  stream_in_row_SA<IN_BIT, SIMD>(
    in, row_buffer, false, storeBufferIdx,IN_W,IN_CH);
  storeBufferIdx++;
  
  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS DEPENDENCE false intra variable=row_buffer
#pragma HLS loop_tripcount min=IN_H - 2 max=IN_H - 2
    // std::cout <<"loadBufferIdx:"<<loadBufferIdx<< endl;
    // std::cout <<"storeBufferIdx:"<<storeBufferIdx<< endl; 
    // std::cout <<"rowIdx: " << rowIdx<< std::endl;   


    stream_in_row_SA<IN_BIT, SIMD>(in, row_buffer, false, storeBufferIdx,IN_W,IN_CH);
    stream_out_data_SA<K, IN_BIT, SIMD>(out, row_buffer, false, rowIdx, loadBufferIdx,IN_H,IN_W,IN_CH,OUTPENUM);

    
    loadBufferIdx++;

    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }

  stream_out_data_SA<K, IN_BIT, SIMD>(out, row_buffer, false, rowIdx, loadBufferIdx,IN_H,IN_W,IN_CH,OUTPENUM);
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  stream_out_data_SA<K, IN_BIT, SIMD>(out, row_buffer, false, rowIdx, loadBufferIdx,IN_H,IN_W,IN_CH,OUTPENUM);

  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}


template <unsigned A_Row, unsigned A_Col,	 unsigned InStreamW,
      unsigned OutStreamW>
void A_to_array(
	stream<ap_uint<InStreamW> > & in,
	stream<ap_uint<OutStreamW> > out[A_Row][A_Col],
  const unsigned NumLines){
  ap_uint<OutStreamW> temp_row;

	for (unsigned long long rep = 0; rep < NumLines; rep++) {
#pragma HLS loop_tripcount min=NumLines max=NumLines
#pragma HLS PIPELINE II=1
		ap_uint<InStreamW> temp = in.read();
    for(unsigned int r = 0; r < A_Row; r++){
      temp_row=temp((r+1)*OutStreamW-1,r*OutStreamW);
      for(unsigned int c = 0; c < A_Col; c++){
        out[r][c].write(temp_row);
      }
    }
	}
}


template <unsigned IN_BIT, unsigned OUT_W,unsigned SIMD, unsigned SIMDNUM>
void buffer_in_row(stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_in, 
                  ap_uint<SIMD * IN_BIT * 2> row_buffer[SIMDNUM][OUT_W/2]){

    for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++){ 
        for (unsigned int w = 0; w < OUT_W /2; w++) {  // OUT_W / 2   80/2
#pragma HLS pipeline II=1  
            ap_uint<SIMD * IN_BIT * 2> data;
            data = fifo_A_in.read();
            row_buffer[infoldIdx][w]=data;
        }
    }     
}

template <unsigned IN_BIT, unsigned OUT_W,unsigned SIMD, unsigned SIMDNUM>
void buffer_out_row(ap_uint<SIMD * IN_BIT * 2> row_buffer[SIMDNUM][OUT_W/2],
                    stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_out){


    for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++){ 
        for (unsigned int w = 0; w < OUT_W /2; w++) {  // OUT_W / 2   80/2
#pragma HLS pipeline II=1  
            ap_uint<SIMD * IN_BIT * 2> data;
            data = row_buffer[infoldIdx][w];
            fifo_A_out.write(data);
        }
    }     
}



template <unsigned K, unsigned IN_BIT, unsigned IN_CH,
          unsigned OUT_W, unsigned OUT_H, unsigned OUT_CH,
          unsigned SIMD, unsigned SIMDNUM,unsigned PENUM>
void local_row_buf(stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_in, 
                  stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_local_out
                  ,const unsigned reps = 1) {

    ap_uint<SIMD * IN_BIT * 2> row_buffer[2][SIMDNUM][OUT_W/2];
    bool arb = 0;

// 先写一行row buffer[0]  
    buffer_in_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(fifo_A_in, row_buffer[0]);

    for (unsigned int h = 0; h < OUT_H* PENUM* K * reps-1; h++) {
#pragma HLS LOOP_TRIPCOUNT max=OUT_H* PENUM* K-1 min=OUT_H* PENUM* K-1
 // 40
        if(arb==0){
           buffer_in_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(fifo_A_in, row_buffer[1]);
           buffer_out_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(row_buffer[0],fifo_A_local_out);
        }
        else{
           buffer_in_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(fifo_A_in, row_buffer[0]);
           buffer_out_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(row_buffer[1],fifo_A_local_out);            
        }
        arb = !arb; 
    }

    buffer_out_row<IN_BIT,OUT_W,SIMD,SIMDNUM>(row_buffer[1],fifo_A_local_out); 
}


template <unsigned K, unsigned IN_BIT, unsigned SIMD>
void A_IO_L1_to_PE(stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_in, stream<ap_uint<IN_BIT * 2> > fifo_A_PE_out[SIMD]
                ,const unsigned NumLines) {
#pragma HLS INLINE OFF


    ap_uint<SIMD * IN_BIT * 2 > reg[SIMD];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete

    ap_uint<IN_BIT*2 > temp;
    // unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

    int flag=1;

#ifdef DATA_DEBUG
FILE* fp = fopen("INBIT_2_reorg_SIMD7.txt", "wb");
#endif

    for (unsigned int i = 0; i < NumLines+SIMD-1; i++) { // 40
//#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2)+SIMD-1 max=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2)+SIMD-1
#pragma HLS PIPELINE II=1
// read 读数
      if(i<NumLines){
        reg[0]=fifo_A_in.read();
      }

#ifdef DEBUG
    if(i==80){
      cout<<"start debug!"<<endl;
    }
    cout<<"开始本次寄存器打印......."<<endl;
    for(int t_i=0;t_i<SIMD;t_i++){
      for(int t_j=0;t_j<SIMD;t_j++){
        cout <<reg[t_i](2*IN_BIT*(t_j+1)-1,2*IN_BIT*t_j) << "\t";
      }
      cout<<"\n";
    }
   cout<<"结束本次寄存器打印......."<<endl;
#endif
// 写数逻辑，高位宽数据右移


      if(i<SIMD-1){  // 0,1,..,SIMD-1
        for(int m=0; m<=SIMD-2;m++){
#pragma HLS loop_tripcount min=0 max=SIMD-1
#pragma HLS UNROLL
          if(m<=i){
            temp=reg[m](2*IN_BIT-1,0);
            fifo_A_PE_out[m].write(temp);
            #ifdef DEBUG
              if(m==7){
                fprintf(fp, "%d\n", int(temp));
              }
            #endif
            reg[m]=reg[m]>>(2*IN_BIT);
          }
        }
      }
      else if(i>=NumLines){ // 处理最后
        for(int m=1; m<=SIMD-1;m++){  // flag=1,2,..,SIMD-1
        #pragma HLS UNROLL
          if(m>=flag){
            temp=reg[m](2*IN_BIT-1,0);
            fifo_A_PE_out[m].write(temp);
            #ifdef DEBUG
              if(m==7){
                fprintf(fp, "%d\n", int(temp));
              }
            #endif

            reg[m]=reg[m]>>(2*IN_BIT);
          }
        }
        flag++;
      }
      else{
        for(unsigned m=0; m<SIMD;m++){
        #pragma HLS UNROLL
          temp=reg[m](2*IN_BIT-1,0);
          fifo_A_PE_out[m].write(temp);
          #ifdef DEBUG
            if(m==7){
              fprintf(fp, "%d\n", int(temp));
            }
          #endif

          reg[m]=reg[m]>>(2*IN_BIT);

        }
      }
// 数据滑窗右移
  	  for(unsigned m=SIMD-1; m>0;m--){
      #pragma HLS UNROLL
		    reg[m]=reg[m-1];
	    }
    }

#ifdef DEBUG
fclose(fp);
#endif

}




/**
 * version1:  variable loop bounds 不可以展开
 * */
/*
template <unsigned K, unsigned IN_BIT, unsigned IN_CH,
          unsigned OUT_W, unsigned OUT_H, unsigned OUT_CH,
          unsigned SIMD, unsigned SIMDNUM,unsigned PENUM>
void A_IO_L1_to_PE(stream<ap_uint<SIMD * IN_BIT * 2> > &fifo_A_in, stream<ap_uint<IN_BIT * 2> > fifo_A_PE_out[SIMD]
                ,const unsigned reps = 1) {
#pragma HLS INLINE OFF


    ap_uint<SIMD * IN_BIT * 2 > reg[SIMD];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete

    ap_uint<IN_BIT*2 > temp;
    unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);
    
    int flag=1;

#ifdef DATA_DEBUG
FILE* fp = fopen("INBIT_2_reorg_SIMD7.txt", "wb");
#endif

    for (unsigned int i = 0; i < total_num * reps+SIMD-1; i++) { // 40
#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2)+SIMD-1 max=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2)+SIMD-1
#pragma HLS PIPELINE II=1 
// read 读数
      if(i<total_num * reps){
        reg[0]=fifo_A_in.read(); 
      }

#ifdef DEBUG
    if(i==80){
      cout<<"start debug!"<<endl;
    }
    cout<<"开始本次寄存器打印......."<<endl;
    for(int t_i=0;t_i<SIMD;t_i++){
      for(int t_j=0;t_j<SIMD;t_j++){
        cout <<reg[t_i](2*IN_BIT*(t_j+1)-1,2*IN_BIT*t_j) << "\t";
      }
      cout<<"\n";
    }
   cout<<"结束本次寄存器打印......."<<endl;
#endif
// 写数逻辑，高位宽数据右移
      if(i<SIMD-1){  // 0,1,..,SIMD-1
        for(int m=0; m<=i;m++){  
#pragma HLS loop_tripcount min=0 max=SIMD-1
#pragma HLS UNROLL
          temp=reg[m](2*IN_BIT-1,0);
          fifo_A_PE_out[m].write(temp);
          #ifdef DEBUG
            if(m==7){
              fprintf(fp, "%d\n", int(temp));
            }
          #endif
          reg[m]=reg[m]>>(2*IN_BIT);
        }        
      }
      else if(i>=total_num * reps){ // 处理最后 
        for(int m=flag; m<=SIMD-1;m++){  // flag=1,2,..,SIMD-1
        #pragma HLS UNROLL	 
          temp=reg[m](2*IN_BIT-1,0);
          fifo_A_PE_out[m].write(temp);
          #ifdef DEBUG
            if(m==7){
              fprintf(fp, "%d\n", int(temp));
            }
          #endif

          reg[m]=reg[m]>>(2*IN_BIT);
        }
        flag++;
      }
      else{
        for(unsigned m=0; m<SIMD;m++){
        #pragma HLS UNROLL	 
          temp=reg[m](2*IN_BIT-1,0);
          fifo_A_PE_out[m].write(temp);
          #ifdef DEBUG
            if(m==7){
              fprintf(fp, "%d\n", int(temp));
            }
          #endif

          reg[m]=reg[m]>>(2*IN_BIT);

        }        
      }
// 数据滑窗右移
  	  for(unsigned m=SIMD-1; m>0;m--){
      #pragma HLS UNROLL	 
		    reg[m]=reg[m-1];
	    }
    }

#ifdef DEBUG
fclose(fp);
#endif

}


*/

// weights[PE][(IN_CH*K / SIMD) * OUT_CH / PE]




template <unsigned K, unsigned IN_CH, unsigned OUT_W, 
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned SIMD, unsigned SIMDNUM,unsigned PE, 
          unsigned PENUM>
void W_IO_L3_in_serialize(const ap_uint<SIMD * K * W_BIT> weights[PE][(IN_CH*K / SIMD) * OUT_CH / PE],
                          stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_local_out, const unsigned reps = 1) {
#pragma HLS INLINE OFF

    ap_uint<SIMD * K *W_BIT>  w;


#ifdef DEBUG
  FILE* fp_win0 = fopen("W3_reorg_SIMD_all_in.txt", "wb");
#endif 
  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
        for (unsigned int infoldIdx = 0; infoldIdx < K* SIMDNUM; infoldIdx++) {
          for (unsigned int p = 0; p < PE; p++) {
#pragma HLS PIPELINE II=1
        // read SIMD*K*4bit
            w=weights[p][peIdx*K*SIMDNUM+infoldIdx];
            fifo_W_local_out.write(w);
          }
        }
      }
    }

#ifdef DEBUG
  fclose(fp_win0);
#endif 

#ifdef DEBUG
  FILE* fp_win = fopen("W3_reorg_PE_0_0.txt", "wb");
/*
  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
          for (unsigned int w = 0; w < OUT_W /2; w++) {
            if(w==0){
              for (unsigned int p = 0; p < PE; p++) {
            // read SIMD*K*4bit
                w0=weights[p][K*k1+0][peIdx*SIMDNUM+infoldIdx];
                w1=weights[p][K*k1+1][peIdx*SIMDNUM+infoldIdx];
                w2=weights[p][K*k1+2][peIdx*SIMDNUM+infoldIdx];    // 这个后面可以改变存储逻辑

                  for(unsigned int m = 0; m < SIMD; m++){
                    ap_int<W_BIT>  d2,d1,d0;
                    d2=w2(W_BIT-1,0);
                    d1=w1(W_BIT-1,0);
                    d0=w0(W_BIT-1,0); 
                    temp_K=(d2,d1,d0);
                    w2=w2>>(W_BIT);
                    w1=w1>>(W_BIT);
                    w0=w0>>(W_BIT);
                    if(p==1){
                    fprintf(fp_win, "%d\n", int(temp_K));
                    }
                  }
              }
            }

          }
        }
      }
    }
  }
*/
  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
          for (unsigned int w = 0; w < OUT_W /2; w++) {
              for (unsigned int p = 0; p < PE; p++) {
            // read SIMD*K*4bit
                w0=weights[p][K*k1+0][peIdx*SIMDNUM+infoldIdx];
                w1=weights[p][K*k1+1][peIdx*SIMDNUM+infoldIdx];
                w2=weights[p][K*k1+2][peIdx*SIMDNUM+infoldIdx];    // 这个后面可以改变存储逻辑

                  for(unsigned int m = 0; m < SIMD; m++){
                    ap_int<W_BIT>  d2,d1,d0;
                    d2=w2(W_BIT-1,0);
                    d1=w1(W_BIT-1,0);
                    d0=w0(W_BIT-1,0); 
                    temp_K=(d2,d1,d0);
                    w2=w2>>(W_BIT);
                    w1=w1>>(W_BIT);
                    w0=w0>>(W_BIT);
                    if(p==1&&m==7){
                    fprintf(fp_win, "%d\n", int(temp_K));
                    }
                  }
              }

          }
        }
      }
    }
  }
  fclose(fp_win);

#endif 

}


template <unsigned K, 
          unsigned W_BIT,
          unsigned SIMD, unsigned PE>
void W_IO_L2_in(int idx, stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_in, stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_out, 
                stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_local_out, 
                const unsigned OUT_H,
                const unsigned PENUM,
                const unsigned SIMDNUM) {
#pragma HLS INLINE OFF
int p0 = idx; // module id

  for (unsigned int h = 0; h < OUT_H; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
            for (unsigned int p = p0; p < PE; p++) {
#pragma HLS PIPELINE II=1  
              if(p==p0){
                  ap_uint<SIMD * K * W_BIT> in_data;
                  in_data = fifo_W_in.read();
                  fifo_W_local_out.write(in_data);
              }
              else{
                  ap_uint<SIMD * K * W_BIT> in_data;
                  in_data = fifo_W_in.read();
                  fifo_W_out.write(in_data);               
              }
            }
        }
      }
    }
  }
}


template <unsigned K, unsigned W_BIT,
          unsigned SIMD, unsigned PE>
void W_IO_L2_in_boundary(int idx, stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_in, stream<ap_uint<SIMD *K * W_BIT> > &fifo_W_local_out,
                        const unsigned OUT_H, const unsigned PENUM, const unsigned SIMDNUM, const unsigned reps = 1) {
#pragma HLS INLINE OFF
int p0 = idx; // module id

  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
            for (unsigned int p = p0; p < PE; p++) {
#pragma HLS PIPELINE II=1
              if(p==p0){
                  ap_uint<K *SIMD * W_BIT> out_data;
                  out_data = fifo_W_in.read();
                  fifo_W_local_out.write(out_data);
              }
            }
        }
      }
    }
  }
}

template <unsigned K, unsigned W_BIT,unsigned SIMD>
void W_IO_L1_in_inter_trans(int idx, ap_uint<K * W_BIT> &local_W, stream<ap_uint<K * W_BIT> > &fifo_W_in, 
                            stream<ap_uint<K * W_BIT> > &fifo_W_out, bool inter_trans_en) {
#pragma HLS INLINE OFF
  int p0 = idx; // module id

  if (!inter_trans_en) return;

  for(unsigned int i = p0; i < SIMD; i++){
#pragma HLS PIPELINE II=1
    if(i==p0){
      ap_uint<K * W_BIT> in_data;
      ap_uint<K * W_BIT> out_data;
      in_data = fifo_W_in.read();
      out_data = in_data;
      local_W=out_data;
    }
    else{
      ap_uint<K * W_BIT> in_data;
      ap_uint<K * W_BIT> out_data;
      in_data = fifo_W_in.read();
      out_data = in_data;
      fifo_W_out.write(out_data);
    }     
  }

}


template <unsigned K, unsigned W_BIT,unsigned SIMD>
void W_IO_L1_in_inter_trans_boundary(int idx, ap_uint<K * W_BIT> &local_W, stream<ap_uint<K * W_BIT> > &fifo_W_in, 
                                     bool inter_trans_en) {
#pragma HLS INLINE OFF
  int p0 = idx; // module id

  if (!inter_trans_en) return;

  for(unsigned int i = p0; i < SIMD; i++){
#pragma HLS PIPELINE II=1
    if(i==p0){
      ap_uint<K * W_BIT> in_data;
      ap_uint<K * W_BIT> out_data;
      in_data = fifo_W_in.read();
      out_data = in_data;
      local_W=out_data;
    }   
  }

}


template <unsigned K, unsigned W_BIT,unsigned OUT_W>
void W_IO_L1_in_intra_trans(ap_uint<K * W_BIT> &local_W, stream<ap_uint<K * W_BIT> > &fifo_W_local_out, bool intra_trans_en) {
#pragma HLS INLINE OFF

  if (!intra_trans_en) return;

    for (unsigned int w = 0; w < OUT_W /2; w++) {
#pragma HLS PIPELINE II=1
      fifo_W_local_out.write(local_W);
    }

}

template <unsigned K, unsigned IN_CH, unsigned OUT_W,
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned SIMD, unsigned SIMDNUM,unsigned PE,
          unsigned PENUM>
void W_IO_L1_to_PE(stream<ap_uint<SIMD * W_BIT * K> > &fifo_W_in, stream<ap_uint<W_BIT * K> > fifo_W_PE_out[SIMD]
                ,const unsigned reps = 1) {
#pragma HLS INLINE OFF


    ap_uint<SIMD * W_BIT * K > reg[SIMD];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete

    ap_uint<W_BIT * K > temp;
    unsigned int total_num=OUT_H*PENUM*K*SIMDNUM;

    int flag=1;

#ifdef DATA_DEBUG
FILE* fp = fopen("INBIT_2_reorg_SIMD7.txt", "wb");
#endif

    for (unsigned int i = 0; i < total_num * reps+SIMD-1; i++) { // 40
#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM+SIMD-1 max=OUT_H*PENUM*K*SIMDNUM+SIMD-1
#pragma HLS PIPELINE II=1
// read 读数
      if(i<total_num * reps){
        reg[0]=fifo_W_in.read();
      }

#ifdef DEBUG
    if(i==80){
      cout<<"start debug!"<<endl;
    }
    cout<<"开始本次寄存器打印......."<<endl;
    for(int t_i=0;t_i<SIMD;t_i++){
      for(int t_j=0;t_j<SIMD;t_j++){
        cout <<reg[t_i](2*IN_BIT*(t_j+1)-1,2*IN_BIT*t_j) << "\t";
      }
      cout<<"\n";
    }
   cout<<"结束本次寄存器打印......."<<endl;
#endif
// 写数逻辑，高位宽数据右移
      if(i<SIMD-1){  // 0,1,..,SIMD-1
        for(int m=0; m<=SIMD-2;m++){
#pragma HLS loop_tripcount min=0 max=SIMD-1
#pragma HLS UNROLL
          if(m<=i){
            temp=reg[m](W_BIT * K-1,0);
            fifo_W_PE_out[m].write(temp);
            #ifdef DEBUG
              if(m==7){
                fprintf(fp, "%d\n", int(temp));
              }
            #endif
            reg[m]=reg[m]>>(W_BIT * K);
          }
        }
      }
      else if(i>=total_num * reps){ // 处理最后
        for(int m=1; m<=SIMD-1;m++){  // flag=1,2,..,SIMD-1
        #pragma HLS UNROLL
          if(m>=flag){
            temp=reg[m](W_BIT * K-1,0);
            fifo_W_PE_out[m].write(temp);
            #ifdef DEBUG
              if(m==7){
                fprintf(fp, "%d\n", int(temp));
              }
            #endif

            reg[m]=reg[m]>>(W_BIT * K);
          }
        }
        flag++;
      }
      else{
        for(unsigned m=0; m<SIMD;m++){
        #pragma HLS UNROLL
          temp=reg[m](W_BIT * K-1,0);
          fifo_W_PE_out[m].write(temp);
          #ifdef DEBUG
            if(m==7){
              fprintf(fp, "%d\n", int(temp));
            }
          #endif

          reg[m]=reg[m]>>(W_BIT * K);

        }
      }
// 数据滑窗右移
  	  for(unsigned m=SIMD-1; m>0;m--){
      #pragma HLS UNROLL
		    reg[m]=reg[m-1];
	    }
    }

#ifdef DEBUG
fclose(fp);
#endif

}


template <unsigned K, unsigned IN_CH, unsigned OUT_W, 
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned SIMD, unsigned SIMDNUM,unsigned PE, 
          unsigned PENUM>
void W_IO_L1_in(int idx, int idy, hls::stream<ap_uint<K * W_BIT> > &fifo_W_in, stream<ap_uint<K * W_BIT> > &fifo_W_out, 
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,const unsigned reps = 1) {
#pragma HLS INLINE OFF

  ap_uint<K * W_BIT> local_W_ping;
  ap_uint<K * W_BIT> local_W_pong;

  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;

  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
            if (arb == 0) {
              W_IO_L1_in_inter_trans<K, W_BIT,SIMD>(
                /* module id */ idy, 
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_in, 
                /* fifo */ fifo_W_out, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );
            }
            else{
              W_IO_L1_in_inter_trans<K, W_BIT,SIMD>(
                /* module id */ idy, 
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_in, 
                /* fifo */ fifo_W_out, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );            
            }
            intra_trans_en = 1;
            arb = !arb; 
          }
      }
    }
  }

  if (arb == 0) {
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_ping, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }
  else{
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_pong, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }

}


template <unsigned K, unsigned W_BIT,
          unsigned SIMD, unsigned PE>
void W_IO_L1_in_WOPP(int idx, int idy, hls::stream<ap_uint<K * W_BIT> > &fifo_W_in, stream<ap_uint<K * W_BIT> > &fifo_W_out, 
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {
#pragma HLS INLINE OFF


  const unsigned int max_cycles = MAX(OUT_W /2,SIMD-idy);
  const unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*max_cycles;

  ap_uint<K * W_BIT> local_W;
  ap_uint<K * W_BIT> local_W_tmp;
  ap_uint<K * W_BIT> in_data;
  ap_uint<K * W_BIT> out_data; 

  unsigned x = 0;
  unsigned y = 0;
  for (unsigned int h = 0; h < total_num  ; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*max_cycles max=OUT_H*PENUM*K*SIMDNUM*max_cycles
#pragma HLS PIPELINE II=1
      if(y<SIMD-idy){
        	in_data = fifo_W_in.read();
      }

			if(y==0){
        local_W=in_data;
			}
			else if(y<SIMD-idy){
				fifo_W_out.write(in_data);
			}

      if(h>=max_cycles & y<OUT_W /2){
        fifo_W_local_out.write(local_W_tmp);
      }

      if(y==max_cycles-1){
        local_W_tmp=local_W;
      }

			if(y==max_cycles-1){
				y=0;
			}
			else{
				y++;
			}

  }

  for(unsigned i = 0; i < OUT_W /2; i++){
#pragma HLS PIPELINE II=1
    fifo_W_local_out.write(local_W_tmp);
  }




}




template <unsigned K, unsigned W_BIT,unsigned SIMD>
void W_IO_L1_in_inter_trans_SIMD(int idx, ap_uint<K * W_BIT> &local_W, stream<ap_uint<SIMD *K * W_BIT> > &fifo_W_in, 
                            stream<ap_uint<K * W_BIT> > &fifo_W_out, bool inter_trans_en) {
#pragma HLS INLINE OFF
  int p0 = idx; // module id

  if (!inter_trans_en) return;

  ap_uint<SIMD * K * W_BIT> in_data;
  ap_uint<K * W_BIT> out_data;
  in_data = fifo_W_in.read();

  for(unsigned int i = p0; i < SIMD; i++){
#pragma HLS PIPELINE II=1
    out_data=in_data(K * W_BIT-1,0);
    if(i==p0){
      local_W=in_data;
    }
    else{
      fifo_W_out.write(out_data);
    } 
    in_data=in_data>>K * W_BIT;    
  }

}


template <unsigned K, unsigned IN_CH, unsigned OUT_W, 
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned SIMD, unsigned SIMDNUM,unsigned PE, 
          unsigned PENUM>
void W_IO_L1_in_SIMD(int idx, int idy, hls::stream<ap_uint<K * SIMD * W_BIT> > &fifo_W_in, stream<ap_uint<K * W_BIT> > &fifo_W_out, 
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,const unsigned reps = 1) {
#pragma HLS INLINE OFF

  ap_uint<K * W_BIT> local_W_ping;
  ap_uint<K * W_BIT> local_W_pong;

  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;

  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
            if (arb == 0) {
              W_IO_L1_in_inter_trans_SIMD<K, W_BIT,SIMD>(
                /* module id */ 0, 
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_in, 
                /* fifo */ fifo_W_out, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );
            }
            else{
              W_IO_L1_in_inter_trans_SIMD<K, W_BIT,SIMD>(
                /* module id */ 0, 
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_in, 
                /* fifo */ fifo_W_out, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );            
            }
            intra_trans_en = 1;
            arb = !arb; 
          }
      }
    }
  }

  if (arb == 0) {
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_ping, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }
  else{
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_pong, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }

}


template <unsigned K, unsigned W_BIT,
          unsigned SIMD, unsigned PE>
void W_IO_L1_in_SIMD_WOPP(int idx, int idy, hls::stream<ap_uint<K * SIMD * W_BIT> > &fifo_W_in, stream<ap_uint<K * W_BIT> > &fifo_W_out, 
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {
#pragma HLS INLINE OFF

  const unsigned int max_cycles = MAX(OUT_W /2,SIMD);
  const unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*max_cycles;

  ap_uint<K * W_BIT> local_W;
  ap_uint<K * W_BIT> local_W_tmp;
  ap_uint<SIMD * K * W_BIT> in_data;
  ap_uint<K * W_BIT> out_data; 

  unsigned x = 0;
  unsigned y = 0;
  for (unsigned int h = 0; h < total_num  ; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*max_cycles max=OUT_H*PENUM*K*SIMDNUM*max_cycles
#pragma HLS PIPELINE II=1

			if(y==0){
				in_data = fifo_W_in.read();
			}

			if(y==0){
				out_data=in_data(K * W_BIT-1,0);
        local_W=out_data;
				in_data=in_data>>K * W_BIT; 
			}
			else if(y<SIMD){
        out_data=in_data(K * W_BIT-1,0);
				fifo_W_out.write(out_data);
				in_data=in_data>>K * W_BIT; 
			}

      if(h>=max_cycles & y<OUT_W /2){
        fifo_W_local_out.write(local_W_tmp);
      }

      if(y==max_cycles-1){
        local_W_tmp=local_W;
      }

			if(y==max_cycles-1){
				y=0;
			}
			else{
				y++;
			}

  }

  for(unsigned i = 0; i < OUT_W /2; i++){
#pragma HLS PIPELINE II=1
    fifo_W_local_out.write(local_W_tmp);
  }


}


template <unsigned K, unsigned IN_CH, unsigned OUT_W, 
          unsigned OUT_H, unsigned OUT_CH, unsigned W_BIT,
          unsigned SIMD, unsigned SIMDNUM,unsigned PE, 
          unsigned PENUM>
void W_IO_L1_in_boundary(int idx, int idy, hls::stream<ap_uint<K * W_BIT> > &fifo_W_in,
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,const unsigned reps = 1) {
#pragma HLS INLINE OFF

  ap_uint<K * W_BIT> local_W_ping;
  ap_uint<K * W_BIT> local_W_pong;

  bool arb = 0;
  bool inter_trans_en = 1;
  bool intra_trans_en = 0;

  for (unsigned int h = 0; h < OUT_H * reps; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int k1 = 0; k1 < K; k1++) {
        for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM; infoldIdx++) {
            if (arb == 0) {
              W_IO_L1_in_inter_trans_boundary<K, W_BIT,SIMD>(
                /* module id */ idy, 
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_in, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );
            }
            else{
              W_IO_L1_in_inter_trans_boundary<K, W_BIT,SIMD>(
                /* module id */ idy, 
                /* array */ local_W_ping, 
                /* fifo */ fifo_W_in, 
                /* enable */ inter_trans_en                
              );
              W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
                /* array */ local_W_pong, 
                /* fifo */ fifo_W_local_out, 
                /* enable */ intra_trans_en
              );            
            }
            intra_trans_en = 1;
            arb = !arb; 
          }
      }
    }
  }

  if (arb == 0) {
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_ping, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }
  else{
    W_IO_L1_in_intra_trans<K, W_BIT,OUT_W>(
      /* array */ local_W_pong, 
      /* fifo */ fifo_W_local_out, 
      /* enable */ intra_trans_en
    );
  }

}


template <unsigned K, unsigned W_BIT,
          unsigned SIMD, unsigned PE>
void W_IO_L1_in_boundary_WOPP(int idx, int idy, hls::stream<ap_uint<K * W_BIT> > &fifo_W_in,
                stream<ap_uint<K * W_BIT> > &fifo_W_local_out,
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {
#pragma HLS INLINE OFF


  const unsigned int max_cycles = MAX(OUT_W /2,SIMD-idy);
  const unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*max_cycles;

  ap_uint<K * W_BIT> local_W;
  ap_uint<K * W_BIT> local_W_tmp;
  ap_uint<K * W_BIT> in_data;
  ap_uint<K * W_BIT> out_data; 

  unsigned x = 0;
  unsigned y = 0;
  for (unsigned int h = 0; h < total_num  ; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*max_cycles max=OUT_H*PENUM*K*SIMDNUM*max_cycles
#pragma HLS PIPELINE II=1
      if(y<SIMD-idy){
        	in_data = fifo_W_in.read();
      }

			if(y==0){
        local_W=in_data;
			}
      
      if(h>=max_cycles & y<OUT_W /2){
        fifo_W_local_out.write(local_W_tmp);
      }

      if(y==max_cycles-1){
        local_W_tmp=local_W;
      }

			if(y==max_cycles-1){
				y=0;
			}
			else{
				y++;
			}

  }

  for(unsigned i = 0; i < OUT_W /2; i++){
#pragma HLS PIPELINE II=1
    fifo_W_local_out.write(local_W_tmp);
  }


}



// 后续再确认这个逻辑对不对 20230307
template <unsigned K, unsigned PROD_BIT>
void C_PE_dummy_out(int idx, stream<ap_uint<PROD_BIT*4> > &fifo_C_out,
                    const unsigned numlines) {

  unsigned int total_num=numlines;

  for (unsigned int h = 0; h < total_num; h++) { // 40
//#pragma HLS loop_tripcount min=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2) max=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2)
#pragma HLS PIPELINE II=1
    ap_uint<PROD_BIT*4> fifo_data = 0;
    fifo_C_out.write(fifo_data);
  }

  // ap_uint<PROD_BIT*4> fifo_data = 0;   // 后续再确认这个逻辑对不对 20230307
  // fifo_C_out.write(fifo_data);
}

template <unsigned K, unsigned IN_BIT, unsigned W_BIT, 
          unsigned PROD_BIT>
void DSP_MULT(ap_uint<K * W_BIT> W_in, ap_uint<2 * IN_BIT> A_in, 
              ap_int<PROD_BIT> &partial0,ap_int<PROD_BIT> &partial1
              ,ap_int<PROD_BIT> &partial2,ap_int<PROD_BIT> &partial3){

  ap_uint<IN_BIT> fifo_data_A0,fifo_data_A1;
  ap_int<W_BIT> fifo_data_W0,fifo_data_W1,fifo_data_W2;
  ap_int<PROD_BIT * 4> m = 0;

  (fifo_data_A1,fifo_data_A0) = A_in;
  (fifo_data_W2,fifo_data_W1,fifo_data_W0) = W_in;
  ap_uint<PROD_BIT+IN_BIT> B_port=(fifo_data_A1, (ap_uint<PROD_BIT - IN_BIT>)0,fifo_data_A0);
  ap_int<PROD_BIT*2+W_BIT> D_port=fifo_data_W0*(1<<PROD_BIT*2)+fifo_data_W1*(1<<PROD_BIT)+fifo_data_W2;




  m= B_port*D_port;

  ap_int<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
  ap_int<PROD_BIT + 1> p1 = m(PROD_BIT * 2 - 1, PROD_BIT - 1);
  ap_int<PROD_BIT + 1> p2 = m(PROD_BIT * 3 - 1, PROD_BIT * 2 - 1);
  ap_int<PROD_BIT + 1> p3 = m(PROD_BIT * 4 - 1, PROD_BIT * 3 - 1); 
  ap_int<PROD_BIT + 1> s1,s2,s3;
  s1=(p1 >> 1) + (p1 & 1);
  s2=(p2 >> 1) + (p2 & 1);
  s3=(p3 >> 1) + (p3 & 1);


  partial0=p0;
  partial1=s1;
  partial2=s2;
  partial3=s3;

#ifdef COM_DEBUG
     cout << "check: "  << endl;
     cout << "m: " << m << endl;
     cout << "partial0: " << partial0 << endl;
     cout << "partial1: " << partial1 << endl;
     cout << "partial2: " << partial2 << endl;
     cout << "partial3: " << partial3 << endl;
     cout << "一次结束！ " << endl;
#endif

}


template <unsigned K, unsigned IN_BIT, unsigned W_BIT,
          unsigned SIMD,  unsigned PE,unsigned PROD_BIT>
void PE_wrapper(int idr, int idc, stream<ap_uint<SIMD * K * W_BIT> > &fifo_W_in, 
                stream<ap_uint<SIMD * 2 * IN_BIT> > &fifo_A_in,
                stream<ap_uint<PROD_BIT*4> > fifo_C_out[PE],
                const unsigned OUT_W,
                const unsigned NumLines) {
#pragma HLS INLINE OFF


    ap_uint<SIMD * IN_BIT * 2 > A_simd_reg[SIMD];
#pragma HLS ARRAY_PARTITION variable=A_simd_reg dim=1 complete
    ap_uint<PE * K * W_BIT > W_pe_reg[PE];
#pragma HLS ARRAY_PARTITION variable=W_pe_reg dim=1 complete

    ap_uint<2*IN_BIT> data_A_reg[SIMD][PE];
#pragma HLS ARRAY_PARTITION variable=data_A_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_A_reg dim=2 complete

    ap_uint<K * W_BIT> data_W_reg[SIMD][PE];
#pragma HLS ARRAY_PARTITION variable=data_W_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_W_reg dim=2 complete

    ap_uint<PROD_BIT*4> data_C_reg[SIMD][PE];
#pragma HLS ARRAY_PARTITION variable=data_C_reg dim=1 complete
#pragma HLS ARRAY_PARTITION variable=data_C_reg dim=2 complete


    int w_index=0;

    // static_assert(PE<=(OUT_W/2), "PE >= OUT_W/2 !!!!");

#ifdef PRINT_PE_DEBUG
      // FILE* fpa0 = fopen("a_r0c0_PE00.txt", "wb");
      // FILE* fpa1 = fopen("a_r0c0_PE10.txt", "wb");
      // FILE* fpa2 = fopen("a_r0c0_PE20.txt", "wb");
      // FILE* fpa3 = fopen("a_r0c0_PE30.txt", "wb");
      FILE* fpw0 = fopen("w_r0c0_PE10.txt", "wb");
      // FILE* fpw1 = fopen("w_r0c0_PE10.txt", "wb");
      // FILE* fpw2 = fopen("w_r0c0_PE20.txt", "wb");
      // FILE* fpw3 = fopen("w_r0c0_PE30.txt", "wb");

      // FILE* fpc0 = fopen("c_r0c0_PE00.txt", "wb");
      // FILE* fpc1 = fopen("c_r0c0_PE10.txt", "wb");
      // FILE* fpc2 = fopen("c_r0c0_PE20.txt", "wb");
      // FILE* fpc3 = fopen("c_r0c0_PE30.txt", "wb");

      // FILE* fp_res30 = fopen("res_r0c0_PE30.txt", "wb");
      // FILE* fp_res31 = fopen("res_r0c0_PE31.txt", "wb");
      // FILE* fp_res32 = fopen("res_r0c0_PE32.txt", "wb");
      // FILE* fp_res33 = fopen("res_r0c0_PE33.txt", "wb");
      // int count=0;
#endif

// #ifdef PRINT_8_DEBUG
//       unsigned w_test=0;
// #endif


  for(unsigned j=0; j<PE;j++){
#pragma HLS UNROLL
    W_pe_reg[j]=0;
  }

  for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL
    A_simd_reg[i]=0;
  }

  for(unsigned j=0; j<PE;j++){
#pragma HLS UNROLL
    for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL
      data_A_reg[i][j]=0;
      data_C_reg[i][j]=0;
      data_W_reg[i][j]=0;
    }
  }


  for (unsigned rep = 0; rep < NumLines+PE+SIMD-2; rep++) { // 40
#pragma HLS PIPELINE II=1

    if(rep<NumLines){
      A_simd_reg[0]=fifo_A_in.read();  // 激活取数后面对不对
    }
    else{
      A_simd_reg[0]=0;
    }

  if( (w_index<PE)&& (rep <NumLines)){
    W_pe_reg[0]=fifo_W_in.read();
  }
  else{
    W_pe_reg[0]=0;
  }



    for(unsigned m=0; m<SIMD;m++){   // 共用一次右移对不对
    #pragma HLS UNROLL
      ap_uint<IN_BIT*2 > a_temp;
      a_temp=A_simd_reg[m](2*IN_BIT-1,0);
      data_A_reg[m][0]=a_temp;
      A_simd_reg[m]=A_simd_reg[m]>>(2*IN_BIT);
    }

// 激活取数后滑窗
    for(unsigned m=SIMD-1; m>0;m--){
    #pragma HLS UNROLL
      A_simd_reg[m]=A_simd_reg[m-1];
    }


    for(unsigned m=0; m<PE;m++){   // 共用一次右移对不对
    #pragma HLS UNROLL
      ap_uint<W_BIT*K> w_temp;
      w_temp=W_pe_reg[m](W_BIT*K-1,0);

      if(   (w_index<PE+SIMD-1)&&(((w_index-m>=0) && (w_index-m<=SIMD-1)) )){
        // data_W_reg[w_index-m][m]=w_temp;
        data_W_reg[w_index-m][m]=w_temp;

        // cout<<"index-simd-"<<m<<"-pe-"<<w_index-m<<"value:"<<w_temp<<endl;
        // cout<<"w_index: "<<w_index<<"  X: "<<w_index-m<<" m:"<<m<<endl;
      }

      if( (OUT_W/2<PE+SIMD-1)&&(w_index+PE-m>=0) && (w_index+PE-m<=SIMD-1) ){

        data_W_reg[w_index-m+PE][m]=w_temp;

        // cout<<"index-simd-"<<m<<"-pe-"<<w_index-m<<"value:"<<w_temp<<endl;
        // cout<<"w_index: "<<w_index<<"  X: "<<w_index-m+PE<<" m:"<<m<<endl;

      }
      W_pe_reg[m]=W_pe_reg[m]>>(W_BIT*K);


    }

    // cout<<"next............"<<endl;

	  for(unsigned m=PE-1; m>0;m--){
	  #pragma HLS UNROLL
		  W_pe_reg[m]=W_pe_reg[m-1];
	  }

      for (int j=PE-1; j>=0;j--) { // PE   // 错了，不是岔开工作，除了前几个时钟和后几个时钟，其他时间所有PE都工作
        for (int i=SIMD-1; i>=0;i--){ // SIMD
#pragma HLS PIPELINE II=1
            // 读数
            ap_uint<2*IN_BIT> data_A_tmp;
            data_A_tmp= data_A_reg[i][j];

            ap_uint<W_BIT*K> data_W_tmp;
            data_W_tmp = data_W_reg[i][j];


            #ifdef PRINT_PE_DEBUG
              if(i==1&&j==0){
                fprintf(fpw0, "%d\n", int(data_W_tmp));
              }
            #endif


            ap_int<PROD_BIT*4> data_C_tmp;
            if(i==0){
                data_C_tmp= 0;  // 上一层的结果
            }
            else{
                data_C_tmp= data_C_reg[i-1][j];
            }
            // PE 计算
            ap_uint<IN_BIT> A0,A1;
            ap_int<W_BIT> W0,W1,W2;
            ap_int<PROD_BIT * 4> result;
            (A1, A0) = data_A_tmp;
            (W2, W1, W0) = data_W_tmp;
            ap_uint<PROD_BIT+IN_BIT> B_port=(A1(IN_BIT-1,0), (ap_uint<PROD_BIT - IN_BIT>)0,A0(IN_BIT-1,0));
            ap_int<PROD_BIT*2+W_BIT> D_port=W0*(1<<PROD_BIT*2)+W1*(1<<PROD_BIT)+W2;

            result=(B_port*D_port)+data_C_tmp;




            // 传递
            if(j<PE-1){
              data_A_reg[i][j+1]=data_A_tmp;
            }
              data_C_reg[i][j]=result;
          }
        }

// 取数逻辑

  for(int d_j=0;d_j<PE;d_j++){
  #pragma HLS UNROLL
   if ( ((SIMD-1+d_j<=rep) && (rep<NumLines) ) || ( (SIMD-1+d_j>(rep-NumLines)) && (rep>=NumLines)) ){   // 这个条件再重新写一下
        fifo_C_out[d_j].write(data_C_reg[SIMD-1][d_j]); 
   }
  }




#ifdef PRINT_8_DEBUG
      // if(rep>=6){
        
      //   if(w_test==OUT_W/2-1){
      //     w_test=0;
      //   }
      //   else{
      //     w_test++;
      //   }
      // } 

#endif


    if(w_index==OUT_W/2-1){
      w_index=0;
    }
    else{
      w_index++;
    }

  }
#ifdef PRINT_PE_DEBUG
  fclose(fpw0);
#endif

}



template <unsigned K, unsigned IN_BIT, unsigned W_BIT,
          unsigned SIMD,  unsigned PE,unsigned PROD_BIT, unsigned M_BIT>
void PE_DSP_ACC(int idr, int idc,  stream<ap_uint<PROD_BIT*4> > fifo_C_in[4], 
                stream<ap_uint<M_BIT*2> > fifo_C_acc[4],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {
#pragma HLS INLINE OFF



ap_int<M_BIT> ACC_P2_prev[PE];
#pragma HLS ARRAY_PARTITION variable=ACC_P2_prev dim=1 complete

ap_int<M_BIT> ACC_P3_prev[PE];
#pragma HLS ARRAY_PARTITION variable=ACC_P3_prev dim=1 complete

ap_int<M_BIT> out0=0;
ap_int<M_BIT> out1=0;
unsigned int Iter_NUM=K*SIMDNUM;

#ifdef IN_DEBUG
      FILE* fpw = fopen("w_40_test.txt", "wb");
      FILE* fpa = fopen("a_40_test.txt", "wb");

#endif

  for (unsigned int h = 0; h < OUT_H; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int peIdx = 0; peIdx < PENUM; peIdx++) {
      for (unsigned int iter = 0; iter < Iter_NUM; iter++) {  
        for (unsigned int w = 0; w < OUT_W /2; w++) {  // OUT_W / 2   80/2
#pragma HLS PIPELINE II=1
            for(unsigned int m=0; m < PE; m++ ){
              bool m_clear = (w == 0);
            // read FM-A
              ap_int<PROD_BIT*4> fifo_data_C;
              fifo_data_C= fifo_C_in[m].read();

              ap_int<PROD_BIT> P0 = fifo_data_C(PROD_BIT - 1, 0);
              ap_int<PROD_BIT + 1> P1 = fifo_data_C(PROD_BIT * 2 - 1, PROD_BIT - 1);
              ap_int<PROD_BIT + 1> P2 = fifo_data_C(PROD_BIT * 3 - 1, PROD_BIT * 2 - 1);
              ap_int<PROD_BIT + 1> P3 = fifo_data_C(PROD_BIT * 4 - 1, PROD_BIT * 3 - 1); 
              ap_int<PROD_BIT + 5> S0, S1,S2,S3;
              S0= P0;
              S1=(P1 >> 1) + (P1 & 1);
              S2=(P2 >> 1) + (P2 & 1);
              S3=(P3 >> 1) + (P3 & 1);

              if (m_clear){ // 1 1
                out0=ACC_P2_prev[m];
                out1=S1;
              }
              else{// 0 1
                out0=S0+ACC_P2_prev[m];
                out1=S1+ACC_P3_prev[m];
              }
              ACC_P2_prev[m]=S2;
              ACC_P3_prev[m]=S3;

              // read 乘累加结果
              fifo_C_acc[m].write((out1,out0));  // 上一层的结果
          }

        }
      }
    }
  }

for(unsigned int m=0; m < PE; m++ ){
  out0=ACC_P2_prev[m];
  fifo_C_acc[m].write((0, out0));
}

#ifdef IN_DEBUG
    if(idx==0&& idy==0 && idr==1&& idc==0){
      fclose(fpw);
      fclose(fpa);
    }
#endif

#ifdef DEBUG
if(idx==1&& idy==7){
  fclose(fp1);
}
#endif

}



template < unsigned IN_BIT>
void A_PE_dummy_in(stream<ap_uint<2 * IN_BIT> > &fifo_A_in, 
                  const unsigned numlines,
                  const unsigned reps = 1) {

  for (unsigned int h = 0; h < numlines * reps; h++) { // 40
#pragma HLS loop_tripcount min=numlines max=numlines
#pragma HLS PIPELINE II=1
      ap_uint<IN_BIT * 2>  in_data = fifo_A_in.read();
    }

}


template <unsigned MAX_A_ROW,unsigned MAX_A_COL,  unsigned PE,unsigned M_BIT>
void arrar_acc_to_Res( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_A_ROW][MAX_A_COL][4], stream<ap_uint<M_BIT*2> > fifo_C_out[PE],
                const unsigned numlines){

ap_uint<M_BIT*2> temp_2m;
ap_uint<M_BIT*2> psum_2m;

ap_int<M_BIT> temp;
ap_int<M_BIT> res;

  for (unsigned int h = 0; h < numlines; h++) { // 40
//#pragma HLS loop_tripcount min=OUT_H max=OUT_H
#pragma HLS PIPELINE II=1
      for(unsigned int c = 0; c < MAX_A_COL; c++){
        for(unsigned int m = 0; m < 4; m++){
          for(unsigned int r = 0; r < MAX_A_ROW; r++){
            if(r==0){
              psum_2m=fifo_C_in[r][c][m].read();
            }
            else{
              temp_2m=fifo_C_in[r][c][m].read();
              for(ap_uint<2> x=0; x<2;x++){
                temp=ap_int<M_BIT>(temp_2m((x+1)*M_BIT-1,x*M_BIT));
                res=ap_int<M_BIT>(psum_2m((x+1)*M_BIT-1,x*M_BIT))+temp;
                psum_2m((x+1)*M_BIT-1,x*M_BIT)=res;
              }                                
            }
          }
          fifo_C_out[c*4+m].write(psum_2m);
        }
    }
  }


      for(unsigned int c = 0; c < MAX_A_COL; c++){
#pragma HLS UNROLL
        for(unsigned int m = 0; m < 4; m++){
#pragma HLS UNROLL
          for(unsigned int r = 0; r < MAX_A_ROW; r++){
#pragma HLS PIPELINE II=1
            if(r==0){
              psum_2m=fifo_C_in[r][c][m].read();
            }
            else{
              temp_2m=fifo_C_in[r][c][m].read();
              for(ap_uint<2> x=0; x<2;x++){
                temp=ap_int<M_BIT>(temp_2m((x+1)*M_BIT-1,x*M_BIT));
                res=ap_int<M_BIT>(psum_2m((x+1)*M_BIT-1,x*M_BIT))+temp;
                psum_2m((x+1)*M_BIT-1,x*M_BIT)=res;
              }                                
            }
          }
          fifo_C_out[c*4+m].write(psum_2m);
        }
    }


}


template <unsigned K, unsigned M_BIT,unsigned MAX_OUP>
void Inter_Reorg_acc_to_Res( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_OUP], 
                stream<ap_uint<M_BIT*2> > fifo_C_out[MAX_OUP],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {

unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

  ap_uint<M_BIT> data0, data1;

  ap_uint<M_BIT> reg[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete
  ap_uint<M_BIT*2> data_in;
  ap_uint<M_BIT*2> data_acc;
  ap_uint<M_BIT*2> res_out;




  ap_uint<2*M_BIT> row_buf[MAX_OUP][MAX_W];
#pragma HLS ARRAY_PARTITION variable=row_buf dim=1 complete

  // ap_uint<2*M_BIT> temp_2m;
  // ap_uint<2*M_BIT> res_2m;
  ap_int<M_BIT> temp0,temp1;
  ap_int<M_BIT> res0,res1;
  ap_int<M_BIT> res_buf0,res_buf1;


  unsigned int w=0;
  unsigned int infoldIdx=0;
  unsigned int outfoldIdx=0;
for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
  (data1, data0) = fifo_C_in[i].read();
  reg[i]=data1;
}

//  for(unsigned int i=0;i<MAX_W;i++){
//#pragma HLS PIPELINE II=1
//    for(unsigned int j=0; j< MAX_OUP;j++){
//    #pragma HLS UNROLL
//        row_buf[j][i]=0;
//    }
//  }


  for (unsigned int h = 0; h < total_num; h++) { // 40
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE false inter variable=row_buf

      for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL  

        (data1, data0) = fifo_C_in[i].read();
        data_in=(data0,reg[i]);
        reg[i]=data1;

        (temp1,temp0)=data_in;

        if(infoldIdx==0){
          data_acc=0;
        }
        else{
          data_acc=row_buf[i][w];
        }

        (res1,res0)=data_acc;


        res_buf0=res0+temp0;
        res_buf1=res1+temp1;

        res_out=(res_buf1,res_buf0);

        if(infoldIdx==SIMDNUM*K-1){
          fifo_C_out[i].write(res_out);
        }

        row_buf[i][w]=res_out;
      }

    if(w==OUT_W/2-1){
      w=0;
      if(infoldIdx==SIMDNUM*K-1){
        infoldIdx=0;
        if(outfoldIdx==PENUM-1){
          outfoldIdx=0;
        }
        else{
          outfoldIdx++;
        }
      }
      else{
        infoldIdx++;
      }
    }
    else{
      w++;
    }
  }
}



/*template <unsigned K, unsigned M_BIT,unsigned MAX_OUP>
void Inter_Reorg_acc_to_Res( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_OUP], 
                stream<ap_uint<M_BIT*2> > fifo_C_out[MAX_OUP],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM) {

unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

  ap_uint<M_BIT> data0, data1;

  ap_uint<M_BIT> reg[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete
  ap_uint<M_BIT*2> data_in;




  ap_uint<2*M_BIT> row_buf[MAX_OUP][MAX_W];
#pragma HLS ARRAY_PARTITION variable=row_buf dim=1 complete

  ap_uint<2*M_BIT> temp_2m;
  ap_uint<2*M_BIT> res_2m;
  ap_int<M_BIT> temp;
  ap_int<M_BIT> res;
  ap_int<M_BIT> res_buf0,res_buf1;


  unsigned int w=0;
  unsigned int infoldIdx=0;
  unsigned int outfoldIdx=0;
for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
  (data1, data0) = fifo_C_in[i].read();
  reg[i]=data1;
}

  for (unsigned int h = 0; h < total_num; h++) { // 40
#pragma HLS PIPELINE II=1
      for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL  


        (data1, data0) = fifo_C_in[i].read();
        data_in=(data0,reg[i]);
        reg[i]=data1;

        if(infoldIdx==0){
          row_buf[i][w]=data_in;
        } 
        else{
          temp_2m=data_in;
          res_2m=row_buf[i][w];

          for(ap_uint<2> m=0; m<2;m++){
            temp=ap_int<M_BIT>(temp_2m((m+1)*M_BIT-1,m*M_BIT));
            res=ap_int<M_BIT>(res_2m((m+1)*M_BIT-1,m*M_BIT))+temp;



            res_2m((m+1)*M_BIT-1,m*M_BIT)=res;



          }  
          row_buf[i][w]=res_2m;        
        }  

        if(infoldIdx==SIMDNUM*K-1){
          ap_uint<2* M_BIT> reg_out; 
          reg_out=row_buf[i][w];
          fifo_C_out[i].write(reg_out);
        }
      }



    if(w==OUT_W/2-1){
      w=0;
      if(infoldIdx==SIMDNUM*K-1){
        infoldIdx=0;
        if(outfoldIdx==PENUM-1){
          outfoldIdx=0;
        }
        else{
          outfoldIdx++;
        }

      }
      else{
        infoldIdx++;
      }
    }
    else{
      w++;
    }

  }


}*/


template <unsigned K, unsigned M_BIT,unsigned I_BIT,unsigned W_BIT,
          unsigned INC_BIT,unsigned BIAS_BIT,unsigned OUT_BIT, unsigned MAX_OUP>
void Inter_Reorg_acc_to_Act( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_OUP], 
                stream<ap_uint<OUT_BIT*2> > fifo_C_out[MAX_OUP], 
                const ap_uint<INC_BIT> inc[MAX_OUP][26],  // 后续改成固定的
	              const ap_uint<BIAS_BIT> bias[MAX_OUP][26],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM,
                const unsigned INCOFFSET) {

unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

  ap_uint<M_BIT> data0, data1;

  ap_uint<M_BIT> reg[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete
  ap_uint<M_BIT*2> data_in;




  ap_uint<2*M_BIT> row_buf[MAX_OUP][MAX_W];
#pragma HLS ARRAY_PARTITION variable=row_buf dim=1 complete

  ap_uint<2*M_BIT> temp_2m;
  ap_uint<2*M_BIT> res_2m;
  ap_int<M_BIT> temp;
  ap_int<M_BIT> res;
  ap_int<M_BIT> res_buf0,res_buf1;
  ap_uint<2*OUT_BIT> outBuf_tmp;
  ap_uint<INC_BIT> inc_tmp;
  ap_uint<BIAS_BIT> bias_tmp;
  unsigned int w=0;
  unsigned int infoldIdx=0;
  unsigned int outfoldIdx=0;
for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
  (data1, data0) = fifo_C_in[i].read();
  reg[i]=data1;
}

  for (unsigned int h = 0; h < total_num; h++) { // 40
#pragma HLS PIPELINE II=1
      for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL  
        (data1, data0) = fifo_C_in[i].read();
        data_in=(data0,reg[i]);
        reg[i]=data1;

        if(infoldIdx==0){
          row_buf[i][w]=data_in;
        } 
        else{
          temp_2m=data_in;
          res_2m=row_buf[i][w];

          for(ap_uint<2> m=0; m<2;m++){
            temp=ap_int<M_BIT>(temp_2m((m+1)*M_BIT-1,m*M_BIT));
            res=ap_int<M_BIT>(res_2m((m+1)*M_BIT-1,m*M_BIT))+temp;
            res_2m((m+1)*M_BIT-1,m*M_BIT)=res;
          }  
          row_buf[i][w]=res_2m;        
        }   

        if(infoldIdx==SIMDNUM*K-1){
          (res_buf1,res_buf0)=row_buf[i][w];
          inc_tmp=inc[i][INCOFFSET+outfoldIdx]; bias_tmp=bias[i][INCOFFSET+outfoldIdx];
          outBuf_tmp(OUT_BIT - 1, 0) =BN_Re15<M_BIT,OUT_BIT,INC_BIT,BIAS_BIT,I_BIT,W_BIT>(res_buf0, inc_tmp, bias_tmp);
          outBuf_tmp(2*OUT_BIT-1,OUT_BIT) =BN_Re15<M_BIT,OUT_BIT,INC_BIT,BIAS_BIT,I_BIT,W_BIT>(res_buf1, inc_tmp, bias_tmp);
          fifo_C_out[i].write(outBuf_tmp);
        }
      }

    if(w==OUT_W/2-1){
      w=0;
      if(infoldIdx==SIMDNUM*K-1){
        infoldIdx=0;
        if(outfoldIdx==PENUM-1){
          outfoldIdx=0;
        }
        else{
          outfoldIdx++;
        }

      }
      else{
        infoldIdx++;
      }
    }
    else{
      w++;
    }

  }


}




/*

template <unsigned K, unsigned M_BIT,unsigned I_BIT,unsigned W_BIT,
          unsigned INC_BIT,unsigned BIAS_BIT,unsigned OUT_BIT, unsigned MAX_OUP>
void Inter_Reorg_acc_to_Res( stream<ap_uint<M_BIT*2> > fifo_C_in[MAX_OUP], 
                stream<ap_uint<OUT_BIT*2> > fifo_C_out[MAX_OUP], 
                const ap_uint<INC_BIT> inc[MAX_OUP][72],  // 后续改成固定的
	              const ap_uint<BIAS_BIT> bias[MAX_OUP][72],
                const unsigned OUT_H,
                const unsigned OUT_W,
                const unsigned PENUM,
                const unsigned SIMDNUM,
                const unsigned INCOFFSET,
                const unsigned layer) {

unsigned int total_num=OUT_H*PENUM*K*SIMDNUM*(OUT_W/2);

  ap_uint<M_BIT> data0, data1;

  ap_uint<M_BIT> reg[MAX_OUP];
#pragma HLS ARRAY_PARTITION variable=reg dim=1 complete
  ap_uint<M_BIT*2> data_in;




  ap_uint<2*M_BIT> row_buf[MAX_OUP][MAX_W/2];
#pragma HLS ARRAY_PARTITION variable=row_buf dim=1 complete

  ap_uint<2*M_BIT> temp_2m;
  ap_uint<2*M_BIT> res_2m;
  ap_int<M_BIT> temp;
  ap_int<M_BIT> res;
  ap_int<M_BIT> res_buf0,res_buf1;
  ap_uint<2*OUT_BIT> outBuf_tmp;
  ap_uint<INC_BIT> inc_tmp;
  ap_uint<BIAS_BIT> bias_tmp;
  
for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL
  (data1, data0) = fifo_C_in[i].read();
  reg[i]=data1;
}

  for (unsigned int h = 0; h < OUT_H; h++) { // 40
#pragma HLS loop_tripcount min=OUT_H max=OUT_H
    for (unsigned int outfoldIdx = 0; outfoldIdx < PENUM; outfoldIdx++) {
      for (unsigned int infoldIdx = 0; infoldIdx < SIMDNUM*K; infoldIdx++) {
          for (unsigned int w = 0; w < OUT_W /2; w++) {
#pragma HLS PIPELINE II=1
          for(unsigned int i=0; i< MAX_OUP;i++){
#pragma HLS UNROLL  
            (data1, data0) = fifo_C_in[i].read();
            data_in=(data0,reg[i]);
            reg[i]=data1;

            if(infoldIdx==0){
              row_buf[i][w]=data_in;
            } 
            else{
              temp_2m=data_in;
              res_2m=row_buf[i][w];

              for(ap_uint<2> m=0; m<2;m++){
                temp=ap_int<M_BIT>(temp_2m((m+1)*M_BIT-1,m*M_BIT));
                res=ap_int<M_BIT>(res_2m((m+1)*M_BIT-1,m*M_BIT))+temp;
                res_2m((m+1)*M_BIT-1,m*M_BIT)=res;
              }  
              row_buf[i][w]=res_2m;        
            }   

            if(infoldIdx==SIMDNUM*K-1){
              (res_buf1,res_buf0)=row_buf[i][w];
              inc_tmp=inc[i][INCOFFSET+outfoldIdx]; bias_tmp=bias[i][INCOFFSET+outfoldIdx];
              outBuf_tmp(OUT_BIT - 1, 0) =BN_Re15_LLSQ<M_BIT,OUT_BIT,INC_BIT,BIAS_BIT,I_BIT,W_BIT>(res_buf0, inc_tmp, bias_tmp,layer);
              outBuf_tmp(2*OUT_BIT-1,OUT_BIT) =BN_Re15_LLSQ<M_BIT,OUT_BIT,INC_BIT,BIAS_BIT,I_BIT,W_BIT>(res_buf1, inc_tmp, bias_tmp,layer);
              fifo_C_out[i].write(outBuf_tmp);
            }

          }

          }
      }
    }
  }

}
*/







template <unsigned PE, unsigned OUT_BIT>
void C_IO_PE_to_out( stream<ap_uint<OUT_BIT*2> > fifo_C_in[PE], 
                stream<ap_uint<PE*OUT_BIT*2> > &fifo_C_out,
                const unsigned NumLines) {

  ap_uint<2* OUT_BIT> reg;  
  ap_uint<PE*OUT_BIT*2> temp;  

  // unsigned int total_num=OUT_H*PENUM*(OUT_W/2);

  for (unsigned int h = 0; h < NumLines; h++) { // 40
//#pragma HLS loop_tripcount min=OUT_H*PENUM*(OUT_W/2) max=OUT_H*PENUM*(OUT_W/2)
#pragma HLS PIPELINE II=1
    for (unsigned int p = 0; p < PE; p++) {
       reg=fifo_C_in[p].read();
       temp((p+1)*OUT_BIT*2-1,p*OUT_BIT*2)=reg;
    }
    fifo_C_out.write(temp);
  }
}





template <unsigned PE, unsigned OUT_BIT>
void DemuxStream2_stride(
      stream<ap_uint<PE*OUT_BIT*2> > &in,
      stream<ap_uint<PE*OUT_BIT*2> > &out_s1,
      stream<ap_uint<PE*OUT_BIT*2> > &out_s2,
      bool stride_flag,
      unsigned NumLines){



	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<PE*OUT_BIT*2> temp = in.read();
		if (stride_flag){
      	out_s1.write(temp);
    }

		else{
      	out_s2.write(temp);
    }

	}

}



template <unsigned PE, unsigned OUT_BIT>
void DemuxStream2_PADPOOL(
      stream<ap_uint<PE*OUT_BIT*2> > &in,
      stream<ap_uint<PE*OUT_BIT*2> > &out_s2_last,
      stream<ap_uint<PE*OUT_BIT*2> > &out_s2_other,
      bool stride_flag,
      bool res_flag,
      unsigned NumLines){

    if(stride_flag)
      return;

	for (unsigned i = 0; i < NumLines; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<PE*OUT_BIT*2> temp = in.read();
		if (res_flag){
      	out_s2_other.write(temp);
    }

		else{
      	out_s2_last.write(temp);
    }

	}

}
