#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>

#include "cudnn.h"
#include "util.h"
#include "Kernel128_winograd.h"


#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

#define MY_KERNEL 1

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] )

__global__ void kernel_128_winograd_BtdB(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
	int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
	int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;

	extern __shared__ float input[];

	int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
	for (int i = 0; i < 6; i++) {
		input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
	}
	__syncthreads();

	float BTd[6];
	switch(Iny1) {
		case 0:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 0, j, Inz)*4 - d(input, 2, j, Inz)*5 + d(input, 4, j, Inz);
			}
			break;
		case 1:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 + d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 2:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 2, j, Inz)*4 - d(input, 3, j, Inz) + d(input, 4, j, Inz);
			}
			break;
		case 3:
			for (int j = 0; j < 6; j++) {
				BTd[j] = -d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) + d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 4:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*2 - d(input, 2, j, Inz) - d(input, 3, j, Inz)*2 + d(input, 4, j, Inz);
			}
			break;
		case 5:
			for (int j = 0; j < 6; j++) {
				BTd[j] = d(input, 1, j, Inz)*4 - d(input, 3, j, Inz)*5 + d(input, 5, j, Inz);
			}
			break;
	}
	__syncthreads();

	int tmp_offset = Iny1*768+Inz;
	for (int i = 0; i < 6; i++) {
		input[tmp_offset + i*stride_c] = BTd[i];
	}
	__syncthreads();

	float BTdB[6];
	switch(Iny1) {
		case 0:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 0, Inz) - 5*d(input, i, 2, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 1:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) + d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 2:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 4*d(input, i, 2, Inz) - d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 3:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = -2*d(input, i, 1, Inz) - d(input, i, 2, Inz) + 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 4:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 2*d(input, i, 1, Inz) - d(input, i, 2, Inz) - 2*d(input, i, 3, Inz) + d(input, i, 4, Inz);
			}
			break;
		case 5:
			for (int i = 0; i < 6; i++) {
				BTdB[i] = 4*d(input, i, 1, Inz) - 5*d(input, i, 3, Inz) + d(input, i, 5, Inz);
			}
			break;
	}
	__syncthreads();

	for (int i = 0; i < 6; i++) {
		pOutputs[(Iny1 + i*6)*2048 + (blockIdx.x*4+blockIdx.y)*128 + Inz] = BTdB[i];
	}
}


__global__ void kernel_128_single_step_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	// kernel_128_single_step_AtIA <<<dim3(4, 4, 128), dim3(4, 4)>>> (ip, l_bnBias, l_bnScale, output);
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Outy = threadIdx.y, kz = blockIdx.z, Outx = threadIdx.x;
	// int c_input = Inx*6 + Iny;

	// __shared__ float bias, scale;
	// extern __shared__ float input[];

	int out_stride_c = 16*128;
	// input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
	// bias = pBiases[kz];
	// scale = pScales[kz];
	// __syncthreads();

	float coeffs[16][36] = {
		{1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, // m = 0, n = 0
		{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0, 2, 2, 2, 2, 2, 0, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0, 0, 0}, // m = 0, n = 1
		{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0}, // m = 0, n = 2
		{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, -1, 0, 8, 8, 8, 8, 8, 0, -8, -8, -8, -8, -8, 0, 1, 1, 1, 1, 1, 0}, // m = 0, n = 3
		{0, 1, -1, 2, -2, 0, 0, 1, -1, 2, -2, 0, 0, 1, -1, 2, -2, 0, 0, 1, -1, 2, -2, 0, 0, 1, -1, 2, -2, 0, 0, 0, 0, 0, 0, 0}, // m = 1, n = 0
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 2, -2, 0, 0, -1, 1, -2, 2, 0, 0, 2, -2, 4, -4, 0, 0, -2, 2, -4, 4, 0, 0, 0, 0, 0, 0, 0}, // m = 1, n = 1
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 2, -2, 0, 0, 1, -1, 2, -2, 0, 0, 4, -4, 8, -8, 0, 0, 4, -4, 8, -8, 0, 0, 0, 0, 0, 0, 0}, // m = 1, n = 2
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 2, -2, 0, 0, -1, 1, -2, 2, 0, 0, 8, -8, 16, -16, 0, 0, -8, 8, -16, 16, 0, 0, 1, -1, 2, -2, 0}, // m = 1, n = 3
		{0, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 0, 0, 0, 0, 0, 0, 0}, // m = 2, n = 0
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 0, 0, -1, -1, -4, -4, 0, 0, 2, 2, 8, 8, 0, 0, -2, -2, -8, -8, 0, 0, 0, 0, 0, 0, 0}, // m = 2, n = 1
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 0, 0, 1, 1, 4, 4, 0, 0, 4, 4, 16, 16, 0, 0, 4, 4, 16, 16, 0, 0, 0, 0, 0, 0, 0}, // m = 2, n = 2
		{0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 4, 0, 0, -1, -1, -4, -4, 0, 0, 8, 8, 32, 32, 0, 0, -8, -8, -32, -32, 0, 0, 1, 1, 4, 4, 0}, // m = 2, n = 3
		{0, 1, -1, 8, -8, 1, 0, 1, -1, 8, -8, 1, 0, 1, -1, 8, -8, 1, 0, 1, -1, 8, -8, 1, 0, 1, -1, 8, -8, 1, 0, 0, 0, 0, 0, 0}, // m = 3, n = 0 
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 8, -8, 1, 0, -1, 1, -8, 8, -1, 0, 2, -2, 16, -16, 2, 0, -2, 2, -16, 16, -2, 0, 0, 0, 0, 0, 0}, // m = 3, n = 1
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 8, -8, 1, 0, 1, -1, 8, -8, 1, 0, 4, -4, 32, -32, 4, 0, 4, -4, 32, -32, 4, 0, 0, 0, 0, 0, 0}, // m = 3, n = 2
		{0, 0, 0, 0, 0, 0, 0, 1, -1, 8, -8, 1, 0, -1, 1, -8, 8, -1, 0, 8, -8, 64, -64, 8, 0, -8, 8, -64, 64, -8, 0, 1, -1, 8, -8, 1}, // m = 3, n = 3
	};
	// float At[4][6] = {{ 1, 1, 1, 1, 1, 0},
	// 				  { 0, 1,-1, 2,-2, 0},
	// 				  { 0, 1, 1, 4, 4, 0},
	// 				  { 0, 1,-1, 8,-8, 1}};
	// float coeffs[1][1] = {{1}};

	int out_coords = Outx*16 + Outy;
	int coeffsIdx = Outx*4 + Outy;
	// coeffs[coeffsIdx][0] * pInputs[0*out_stride_c + glb_out_idx] +
	// int coeffsIdx = 0;
	int glb_out_idx = (Tilex*4 + Tiley)*128 + kz;	
	// pOutputs[coeffsIdx + kz] = 
	// 	At[Outx][0] * At[Outy][0] * pInputs[0*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][0] * pInputs[1*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][0] * pInputs[2*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][0] * pInputs[3*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][0] * pInputs[4*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][0] * pInputs[5*out_stride_c + glb_out_idx] +
	// 	At[Outx][0] * At[Outy][1] * pInputs[6*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][1] * pInputs[7*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][1] * pInputs[8*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][1] * pInputs[9*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][1] * pInputs[10*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][1] * pInputs[11*out_stride_c + glb_out_idx] +
	// 	At[Outx][0] * At[Outy][2] * pInputs[12*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][2] * pInputs[14*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][2] * pInputs[14*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][2] * pInputs[15*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][2] * pInputs[16*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][2] * pInputs[17*out_stride_c + glb_out_idx] +
	// 	At[Outx][0] * At[Outy][3] * pInputs[18*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][3] * pInputs[19*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][3] * pInputs[20*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][3] * pInputs[21*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][3] * pInputs[22*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][3] * pInputs[23*out_stride_c + glb_out_idx] +
	// 	At[Outx][0] * At[Outy][4] * pInputs[24*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][4] * pInputs[25*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][4] * pInputs[26*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][4] * pInputs[27*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][4] * pInputs[28*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][4] * pInputs[29*out_stride_c + glb_out_idx] +
	// 	At[Outx][0] * At[Outy][5] * pInputs[30*out_stride_c + glb_out_idx] +
	// 	At[Outx][1] * At[Outy][5] * pInputs[31*out_stride_c + glb_out_idx] +
	// 	At[Outx][2] * At[Outy][5] * pInputs[32*out_stride_c + glb_out_idx] +
	// 	At[Outx][3] * At[Outy][5] * pInputs[33*out_stride_c + glb_out_idx] +
	// 	At[Outx][4] * At[Outy][5] * pInputs[34*out_stride_c + glb_out_idx] +
	// 	At[Outx][5] * At[Outy][5] * pInputs[35*out_stride_c + glb_out_idx];
	pOutputs[(((Tilex<<2)+1+Outx)*16 + (Tiley<<2)+1+Outy)*128 + kz] =
		coeffs[coeffsIdx][0] * pInputs[0*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][1] * pInputs[1*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][2] * pInputs[2*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][3] * pInputs[3*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][4] * pInputs[4*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][5] * pInputs[5*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][6] * pInputs[6*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][7] * pInputs[7*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][8] * pInputs[8*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][9] * pInputs[9*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][10] * pInputs[10*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][11] * pInputs[11*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][12] * pInputs[12*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][14] * pInputs[14*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][14] * pInputs[14*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][15] * pInputs[15*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][16] * pInputs[16*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][17] * pInputs[17*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][18] * pInputs[18*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][19] * pInputs[19*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][20] * pInputs[20*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][21] * pInputs[21*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][22] * pInputs[22*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][23] * pInputs[23*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][24] * pInputs[24*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][25] * pInputs[25*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][26] * pInputs[26*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][27] * pInputs[27*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][28] * pInputs[28*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][29] * pInputs[29*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][30] * pInputs[30*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][31] * pInputs[31*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][32] * pInputs[32*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][33] * pInputs[33*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][34] * pInputs[34*out_stride_c + glb_out_idx] +
		coeffs[coeffsIdx][35] * pInputs[35*out_stride_c + glb_out_idx];
	// int tmp = c;
	// for (int i=0; i < 36; ++i) {
	// 	tmp += coeffs[coeffsIdx][0] * pInputs[i*out_stride_c + glb_out_idx];
	// }
	// pOutputs[coeffsIdx + kz] = tmp;
}

__global__ void kernel_128_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
	int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
	int c_input = Inx*6 + Iny;

	__shared__ float bias, scale;
	extern __shared__ float input[];

	input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
	bias = pBiases[kz];
	scale = pScales[kz];
	__syncthreads();

	float tmp = 0;
	switch(Inx) {
		case 0:
			tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
			break;
		case 1:
			tmp = input[6+Iny] - input[12+Iny] + 2*input[18+Iny] - 2*input[24+Iny];
			break;
		case 2:
			tmp = input[6+Iny] + input[12+Iny] + 4*input[18+Iny] + 4*input[24+Iny];
			break;
		case 3:
			tmp = input[6+Iny] - input[12+Iny] + 8*input[18+Iny] - 8*input[24+Iny] + input[30+Iny];
			break;
	}
	__syncthreads();

	input[c_input] = tmp;
	__syncthreads();

	if (Inx > 3 || (Tilex == 3 && Inx > 1)) return;
	
	int x;
	float o;
	switch(Iny) {
		case 0:
			x = Inx*6;
			o = 1*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4])+ 0;
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*128 + kz] = o;
			break;
		case 1:
			x = Inx*6;
			o = 1*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + 0;
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*128 + kz] = o;
			break;
		case 2:
			if (Tiley == 3) break;
			x = Inx*6;
			o = 1*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + 0;
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*128 + kz] = o;
			break;
		case 3:
			if (Tiley == 3) break;
			x = Inx*6;
			o = 1*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + 0;
			pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*128 + kz] = o;
			break;
	}
}


__global__ void kernel_128_OuterProduct_128(float *A, float *B, float *C) {
	int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
	int c_input = tY*128 + tX, c_kernel = c_input, T_offset = (Tile<<11) + (Part<<10) + c_input, B_offset = (Tile<<14) + c_kernel;
	
	extern __shared__ float input[];
	float *kernel = input + 1024, *out = kernel + 8192;
	int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};//, 4096, 4224, 4352, 4480, 4608, 4736, 4864, 4992, 5120, 5248, 5376, 5504, 5632, 5760, 5888, 6016, 6144, 6272, 6400, 6528, 6656, 6784, 6912, 7040, 7168, 7296, 7424, 7552, 7680, 7808, 7936, 8064};
	out[c_input] = 0.0f;

	input[c_input] = A[T_offset];

	for (int k = 0; k < 4; k++) {
		int B_start = B_offset + (k<<12); // 32*64
		kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
		kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
		__syncthreads();

		float sum = 0;
		int y_tmp = (tY<<7)+(k<<5);
		for (int j = 0; j < 32; j++) {
			sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
		}
		out[tY*128 + tX] += sum;
		__syncthreads();
	}

	C[T_offset] = out[c_input];
}

int kernel_128() {
	float *input_ = get_parameter(inputName128, 16*16*128);
	float *bias = get_parameter(biasName128, 128);
	float *input, *output, *l_weights, *l_bias, *pooling_output;
	uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
	cudaError_t s;
	cudnnStatus_t status;

	float one = 1.0, zero = 0.0;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////


	/*  1. Data preparation  */
	float *t_input, *ip;
	//float *kernel = get_Winograd_Kernel128(weight_winograd_Name128, 128);
	float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);
	float *l_bnBias, *l_bnScale, *bnBias, *bnScale;

	int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nBias = 128, nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128, nPoolingOutput=9*9*128;
	cudaMalloc((void **) &input, nInput<<3);
	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &pooling_output, nPoolingOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &l_bias, nBias<<2);
	cudaMalloc((void **) &t_input, nTransInput<<2);
	cudaMalloc((void **) &ip, nInnerProd<<2);
	cudaMemset((void *) input, 0, nInput<<3);
	cudaMemset((void *) output, 0, nOutput<<2);
	cudaMemset((void *) t_input, 0, nTransInput<<2);
	cudaMemset((void *) l_weights, 0, nWeights<<2);
	cudaMemset((void *) ip, 0, nInnerProd<<2);
	cudaMemcpy(input, input_, nInput<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bias, bias, nBias<<2, cudaMemcpyHostToDevice);
	
	bnBias = get_parameter(bnBias_winograd_Name128, 128);
	bnScale = get_parameter(bnScale_winograd_Name128, 128);
	cudaMalloc((void **) &l_bnBias, nBias<<2);
	cudaMalloc((void **) &l_bnScale, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);
	float tmp_winograd[nOutput];
	float tmp_winograd_pooled[nPoolingOutput];

	cudnnHandle_t win_handle;
	cudnnTensorDescriptor_t winydesc, winpooldesc;
	status = cudnnCreate(&win_handle);
	cudnnPoolingDescriptor_t winpoolingDesc;
	status = cudnnCreatePoolingDescriptor(&winpoolingDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed16\n");
	// CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
	status = cudnnSetPooling2dDescriptor(winpoolingDesc, CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed17\n");

	status = cudnnCreateTensorDescriptor(&winpooldesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5.5\n");
	status = cudnnSetTensor4dDescriptor(winpooldesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 9, 9);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5.51\n");

	status = cudnnCreateTensorDescriptor(&winydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(winydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 16, 16);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	
	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();

	kernel_128_winograd_BtdB <<<dim3(4, 4), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
	kernel_128_OuterProduct_128<<<dim3(36, 2), dim3(128, 8), (8*128 + 64*128 + 8*128)<<2 >>> (t_input, l_weights, ip);
	kernel_128_winograd_AtIA <<<dim3(4, 4, 128), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
	// kernel_128_single_step_AtIA <<<dim3(4, 4, 128), dim3(4, 4)>>> (ip, l_bnBias, l_bnScale, output);
	//cudaCheckError();
	// status = cudnnPoolingForward(win_handle, winpoolingDesc, &one,
	// 	winydesc, output, &zero,
	// 	winpooldesc, pooling_output);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed4\n");
	cudaDeviceSynchronize();
	
	nT2 = getTimeMicroseconds64();
	printf("TotalTime = %d us\n", nT2-nT1); 


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp_winograd, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	s = cudaMemcpy(tmp_winograd_pooled, pooling_output, nPoolingOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	//cudaCheckError();
	make_file("./tensors/winograd_out.bin", nOutput, tmp_winograd);
	make_file("./tensors/winograd_out_pooled.bin", nPoolingOutput, tmp_winograd_pooled);

	cudaFree(t_input);
	cudaFree(output);
	cudaFree(pooling_output);
	cudaFree(l_weights);
	cudaFree(l_bias);
	cudaFree(ip);

	free(kernel);
	free(bnScale);
	free(bnBias);

	status = cudnnDestroy(win_handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed16\n");


	/////////////////////////////////

	// cuDNN

	/////////////////////////////////

	/*  1. Data preparation  */
	kernel = get_parameter(weight_NCHW_Name128, 9*128*128);
	bnBias = get_parameter(bnBiasName128, 128);
	bnScale = get_parameter(bnScaleName128, 128);
	float* eMean = get_parameter(eMeanName128, 128);
	float* eVar = get_parameter(eVarName128, 128);
	float *l_eMean, *l_eVar;
	nInput = 16*16*128, nOutput = 14*14*128, nWeights = 3*3*128*128, nBias = 128, nPoolingOutput=7*7*128;

	cudaMalloc((void **) &output, nOutput<<2);
	cudaMalloc((void **) &pooling_output, nPoolingOutput<<2);
	cudaMalloc((void **) &l_weights, nWeights<<2);
	cudaMalloc((void **) &l_bias, nBias<<2);
	cudaMemcpy(l_weights, kernel, nWeights<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bias, bias, nBias<<2, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &l_eMean, nBias<<2);
	cudaMalloc((void **) &l_eVar, nBias<<2);
	cudaMemcpy(l_bnBias, bnBias, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_bnScale, bnScale, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eMean, eMean, nBias<<2, cudaMemcpyHostToDevice);
	cudaMemcpy(l_eVar, eVar, nBias<<2, cudaMemcpyHostToDevice);

	cudaMemset((void *) output, 0, nOutput<<2);

	float tmp_cudnn[nOutput];
	float tmp_pooled[nPoolingOutput];


	/*  2. cuDNN preparation  */
	int size;

	cudnnHandle_t handle;
	status = cudnnCreate(&handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

	cudnnTensorDescriptor_t xdesc, ydesc, bdesc, pooldesc;
	cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
	status = cudnnCreateTensorDescriptor(&xdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
	status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 16, 16);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
	status = cudnnCreateTensorDescriptor(&ydesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
	status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 14, 14);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
	status = cudnnCreateTensorDescriptor(&pooldesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5.5\n");
	status = cudnnSetTensor4dDescriptor(pooldesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 7, 7);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed5.51\n");
	status = cudnnCreateFilterDescriptor(&wdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed6\n");
	status = cudnnSetFilter4dDescriptor(wdesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 128, 128, 3, 3);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed7\n");
	status = cudnnCreateTensorDescriptor(&bdesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed8\n");
	status = cudnnSetTensor4dDescriptor(bdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed9\n");
	cudnnConvolutionDescriptor_t conv_desc;
	status = cudnnCreateConvolutionDescriptor(&conv_desc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed10\n");
	status = cudnnSetConvolution2dDescriptor(conv_desc, 0,0, 1,1,1,1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT); //CUDNN_CONVOLUTION
	if (status != CUDNN_STATUS_SUCCESS) printf("failed11\n");

	cudnnActivationDescriptor_t act_desc;
	status = cudnnCreateActivationDescriptor(&act_desc);  
	if (status != CUDNN_STATUS_SUCCESS) printf("failed12\n");
	status = cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed13\n");
	cudnnPoolingDescriptor_t poolingDesc;
	status = cudnnCreatePoolingDescriptor(&poolingDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed16\n");
	// CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
	status = cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX,
		CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed17\n");
	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc;
	status = cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed14\n");
	status = cudnnSetTensor4dDescriptor(bnScaleBiasMeanVarDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 128, 1, 1);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed15\n");


	cudnnConvolutionFwdAlgo_t algo = (cudnnConvolutionFwdAlgo_t)6;

	status = cudnnGetConvolutionForwardWorkspaceSize(handle,
	   xdesc,
	   wdesc,
	   conv_desc,
	   ydesc,
	   algo,
	   (size_t *)&(size));

	float *extra;
	cudaMalloc((void **) &extra, size);


	/*  3. Computing  */
	nT1_cudnn = getTimeMicroseconds64();

	status = cudnnConvolutionForward(handle, &one,
		xdesc, input, wdesc, l_weights, 
		conv_desc, algo, 
		extra, size, &zero,
		ydesc, output);
	if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed1\n");

	// status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
	// 	&one, &zero, 
	// 	ydesc, output, ydesc, output,
	// 	bnScaleBiasMeanVarDesc, l_bnScale, l_bnBias, l_eMean, l_eVar, CUDNN_BN_MIN_EPSILON);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed2\n");

	// status = cudnnActivationForward(handle, act_desc, &one,
	// 	ydesc, output, &zero,
	// 	ydesc, output);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed3\n");

	// status = cudnnPoolingForward(handle, poolingDesc, &one,
	// 	ydesc, output, &zero,
	// 	pooldesc, pooling_output);
	// if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed4\n");

	cudaDeviceSynchronize();
	nT2_cudnn = getTimeMicroseconds64();
	printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);


	/*  4. Copy back and free  */
	s = cudaMemcpy(tmp_cudnn, output, nOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));

	s = cudaMemcpy(tmp_pooled, pooling_output, nPoolingOutput<<2, cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));

	make_file("./tensors/pooled.bin", nPoolingOutput, tmp_pooled);
	make_file("./tensors/cudnnout.bin", nOutput, tmp_cudnn);

	cudaFree(extra);
	cudaFree(input);
	cudaFree(output);
	cudaFree(pooling_output);
	cudaFree(l_weights);
	cudaFree(l_bias);

	cudaFree(l_bnScale);
	cudaFree(l_bnBias);
	cudaFree(l_eMean);
	cudaFree(l_eVar);

	free(bias);
	free(kernel);

	free(bnScale);
	free(bnBias);
	free(eMean);
	free(eVar);
	free(input_);
	status = cudnnDestroy(handle);
	if (status != CUDNN_STATUS_SUCCESS) printf("failed16\n");

	output_checker(tmp_winograd, tmp_cudnn, 14, 128, 1);

	return ((nT2-nT1) << 16) | (nT2_cudnn-nT1_cudnn);
}