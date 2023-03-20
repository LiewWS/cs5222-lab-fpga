#include <stdio.h>
#include <stdlib.h>

#include "mmult.h"

// --------------------------------------------------------------------
// function to be accelerated in HW wrapped with AXI4-Stream interface
void mmult_hw (AXI_VAL in_stream[IS_SIZE], AXI_VAL out_stream[OS_SIZE])
{
#pragma HLS INTERFACE s_axilite port=return     bundle=CONTROL_BUS
#pragma HLS INTERFACE axis      port=in_stream
#pragma HLS INTERFACE axis      port=out_stream

	// Assertions (to avoid out of array bound writes)
	assert(BATCH%TILING==0);
	assert(FEAT%W_WIDTH_RATIO==0);
	assert(FEAT%IN_WIDTH_RATIO==0);
	assert((BATCH*CLASSES)%OUT_WIDTH_RATIO==0);

	// Hardware memory buffers
	out_T offset_buf[CLASSES];
	w_T weight_buf[CLASSES][FEAT];
	in_T in_buf[TILING][FEAT];
	out_T out_buf[TILING][CLASSES];

#pragma HLS ARRAY_PARTITION variable=in_buf factor=128 block dim=2
#pragma HLS ARRAY_PARTITION variable=weight_buf factor=128 block dim=2
#pragma HLS RESOURCE variable=in_buf core=RAM_2P_LUTRAM
#pragma HLS RESOURCE variable=weight_buf core=RAM_2P_BRAM

	// Input and output AXI stream indices
	int is_idx = 0;
	int os_idx = 0;

	// Stream in offset vector
	// CSE548 TODO
	LOAD_OFF: for (int i = 0; i < CLASSES; i += OUT_WIDTH_RATIO) {
		axi_T packet = pop_stream(in_stream[is_idx++]);
		UNPACK_OFF: for (int j = 0; j < OUT_WIDTH_RATIO; ++j) {
			out_bit_T bits = packet >> (j*(OUT_WIDTH));
			offset_buf[i+j] = *((out_T*) &bits) & ((1ULL<<OUT_WIDTH)-1);
		}
	}

	// Stream in weight matrix
	// CSE548 TODO
	LOAD_W_0: for (int i = 0; i < CLASSES; ++i) {
		LOAD_W_1: for (int j = 0; j < FEAT; j += W_WIDTH_RATIO) {
			axi_T packet = pop_stream(in_stream[is_idx++]);
			LOAD_W_BITS: for (int k = 0; k < W_WIDTH_RATIO; ++k) {
				w_bit_T bits = packet >> (k*(W_WIDTH));
				weight_buf[i][j+k] = *((w_T*) &bits) & ((1ULL<<W_WIDTH)-1);
			}
		}
	}

	// Iterate over tiles
	LT: for (int t = 0; t < BATCH; t+=TILING) {

		// Stream in input tile
		// CSE548 TODO
		LOAD_I_0: for (int i = 0; i < TILING; ++i) {
			LOAD_I_1: for (int j = 0; j < FEAT; j += IN_WIDTH_RATIO) {
#pragma HLS UNROLL factor=128
				axi_T packet = pop_stream(in_stream[is_idx++]);
				LOAD_I_BITS: for (int k = 0; k < IN_WIDTH_RATIO; ++k) {
#pragma HLS UNROLL
					in_bit_T bits = packet >> (k*(IN_WIDTH));
					in_buf[i][j+k] = *((in_T*) &bits) & ((1ULL<<IN_WIDTH)-1);
				}
			}
		}

		// Perform matrix multiplication
		L1: for (int i = 0; i < TILING; i++) {
			// Iterate over output classes
			L2: for (int j = 0; j < CLASSES; j++) {
				// Perform the dot product
#pragma HLS PIPELINE II=1
				out_T tmp = offset_buf[j];
				L3: for(int k = 0; k < FEAT; k++) {
#pragma HLS UNROLL
					out_T mult = in_buf[i][k] * weight_buf[j][k];
					tmp += mult;
				}
				out_buf[i][j] = tmp;
			}
		}

		// Stream out output matrix
		// CSE548 TODO
		STORE_0: for (int i = 0; i < TILING; ++i) {
			STORE_1: for (int j = 0; j < CLASSES; j += OUT_WIDTH_RATIO) {
#pragma HLS UNROLL factor=128
				axi_T packet = 0;
				PACK_OUT: for (int k = 0; k < OUT_WIDTH_RATIO; ++k) {
#pragma HLS UNROLL
					out_bit_T bits = *((out_bit_T*) &out_buf[i][j+k]);
					packet |= (bits & ((1ULL<<OUT_WIDTH)-1))<<(k*OUT_WIDTH);
				}
				out_stream[os_idx++] = push_stream(packet, os_idx == (OS_SIZE));
			}
		}
	}
}


// --------------------------------------------------------
// functions to insert and extract elements from an axi stream
// includes conversion to correct data type
axi_T pop_stream(AXI_VAL const &e)
{
#pragma HLS INLINE

	axi_T ret = e.data;

	volatile ap_uint<sizeof(axi_T)> strb = e.strb;
	volatile ap_uint<sizeof(axi_T)> keep = e.keep;
	volatile ap_uint<AXI_U> user = e.user;
	volatile ap_uint<1> last = e.last;
	volatile ap_uint<AXI_TI> id = e.id;
	volatile ap_uint<AXI_TD> dest = e.dest;

	return ret;
}

AXI_VAL push_stream(axi_T const &v, bool last = false)
{
#pragma HLS INLINE

	AXI_VAL e;

	e.data = v;
	e.strb = (1<<sizeof(axi_T))-1;
	e.keep = (1<<sizeof(axi_T))-1;
	e.user = 0;
	e.last = last ? 1 : 0;
	e.id = 0;
	e.dest = 0;
	return e;
}

