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

#define d(input, i, j, Inz) ( input[Inz + i*768 + (j<<7)] ) // as expected: i: row, j: column and Inz: channel in input arr

__global__ void kernel_128_winograd_BtdB(float *pInputs, float *pOutputs) {
    // lunches with (4, 4) blocks and (128, 6) threads per block
    int Inx = blockIdx.x<<2, Iny0 = blockIdx.y<<2, Iny1 = threadIdx.y, Inz = threadIdx.x;
    // Inx : in {0, 4, 8, 12}, row of the target 6*6 grid from input
    // Iny0: column of pInputs like Inx
    // Iny1: our current column the selected grid from 0 to 5
    // Inz: current index in z direction from 0 to 127
    int Iny = Iny0+Iny1, stride_r = 2048, stride_c = 128; // 2048 = 16*128
    // Iny: current column as each thread is responsible for and column of a 6*6 grid. some columns are selected in two threads which is expected
    // stride_r: stride of rows, meaning adding this takes you two the next row (bigger number takes you down)
    // stride_c: above but for columns
    int c_glb_start = Inx*stride_r + Iny*stride_c + Inz, c_input = Iny1*stride_c + Inz;
    // c_glb_start: offset required to get to 6 long column for this thread:
    // Inx selectes row, Iny selectes column and Inz selectes channel, will always end up on top of a 6 long column
    // and will end up on top of all of them. (some of them twice)
    // c_input does the same as above but within shared memory (desired row is always 0'th)

    // shared memory which 6*6*128 floats has allocated for per block 
    // ( each 6*6 grid of input get a different input array in shared memory)
    extern __shared__ float input[];

    int tmp[6] = {0, 768, 1536, 2304, 3072, 3840}; // 768 = 6*128
    for (int i = 0; i < 6; i++) {
        input[c_input + tmp[i]] = pInputs[c_glb_start + i*stride_r];
    }
    __syncthreads();
    // 3 lines above bring the input into shared memroy
    // other threads within block will use what other threads bring, so __syncthreads() 
    // is called.
    // tmp[i] selectes right row, aka: i takes you up and down so one column is brought by each thread

    float BTd[6]; // a row of BTd which requires multiple columns to calculate (hence previous syncthreads call)
    switch(Iny1) { 
        // in previous step Iny1 chose which column this thread brought into shared memory
        // but now it selectes which row of BTd this thread will calculate, hence this thread will only use one row
        // of B, hence one "formulla" needs to be selected.
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

    // write the result calculated above to input in shared memory so other threads can use it
    // the sync threads above is neccessry because we don't want to override something in input that is stll needed above
    // the sync threads below is neccessary becuase this threads needs other rows of Btd
    int tmp_offset = Iny1*768+Inz;
    for (int i = 0; i < 6; i++) {
        input[tmp_offset + i*stride_c] = BTd[i];
    }
    __syncthreads();

    float BTdB[6];
    // Btdb here is one column of BtdB. calculation is same as above except second matrix is transposed
    // (i and j are swapped) meaning we are calculating BT.(BTd)T = (BTd.B)T
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
        // Iny1*2048 selects column, i*6*2048 selects row so changing i takes you up and down
        // the rest selects channnel
        // last syncthreads seems unnecessary since no access to shared memory is done after it.
    }
}


__global__ void kernel_128_winograd_AtIA(float *pInputs, float *pBiases, float *pScales, float *pOutputs) {
    // kernel_128_winograd_AtIA <<<dim3(4, 4, 128), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
    // pInputs is 16 * 128 * 36 aka 6 * 6 * 128 * 16, pBiases is 128 like pSclaes and pOuputs is 16 * 16 * 128
    int Tilex = blockIdx.x, Tiley = blockIdx.y, Iny = threadIdx.y, kz = blockIdx.z, Inx = threadIdx.x;
    // a block for each tile of input so Tilex and Tiley make sense especially becuase the ouput needs to be 16 * 16
    // every thing else is also self-explantory
    int c_input = Inx*6 + Iny;
    // Inx is row, Iny is column

    __shared__ float bias, scale;
    extern __shared__ float input[];
    // each block gets a 6 * 6 input and shared memory for it and also that many threads

    input[c_input] = pInputs[c_input*16*128 + (Tilex*4+Tiley)*128 + kz];
    // bring input to shared memory also NHWC indexing stuff
    bias = pBiases[kz];
    scale = pScales[kz];
    __syncthreads();

    float tmp = 0;
    switch(Inx) {
        case 0:
            tmp = input[Iny] + input[6+Iny] + input[12+Iny] + input[18+Iny] + input[24+Iny];
            // multiply a row of A by dicided by Inx by a column of input decided by Iny
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
    // don't write edges?? yes. also Inx > 3 we don't need since A is 4 rows
    
    int x;
    float o;
    switch(Iny) {
        case 0:
            x = Inx*6;
            o = scale*(input[x]+input[x+1]+input[x+2]+input[x+3]+input[x+4])+ bias;
            pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+1)*128 + kz] = o > 0 ? o : 0;
            // shift by one when writing so combined with not writing edges we write to the middle of the ouput??
            break;
        case 1:
            x = Inx*6;
            o = scale*(input[x+1] - input[x+2] + 2*input[x+3] - 2*input[x+4]) + bias;
            pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+2)*128 + kz] = o > 0 ? o : 0;
            // Tilex and Inx: row, Tiley: column, kz: channel
            break;
        case 2:
            if (Tiley == 3) break;
            // don't write edges
            x = Inx*6;
            o = scale*(input[x+1] + input[x+2] + 4*input[x+3] + 4*input[x+4]) + bias;
            pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+3)*128 + kz] = o > 0 ? o : 0;
            break;
        case 3:
            if (Tiley == 3) break;
            // don't write edges
            x = Inx*6;
            o = scale*(input[x+1] - input[x+2] + 8*input[x+3] - 8*input[x+4] + input[x+5]) + bias;
            pOutputs[(((Tilex<<2)+1+Inx)*16 + (Tiley<<2)+4)*128 + kz] = o > 0 ? o : 0;
            break;
    }
}


__global__ void kernel_128_OuterProduct_128(float *A, float *B, float *C) {
    // kernel_128_OuterProduct_128<<<dim3(36, 2), dim3(128, 8), (8*128 + 64*128 + 8*128)<<2 >>> (t_input, l_weights, ip);
    // A or t_input is 16 * 6 * 6 * 128, B or transformed weights is 6 * 6 * 128 * 128 and finally C or ip is 16 * 6 * 6 * 128
    // there is a thread for each single output
    int Tile = blockIdx.x, Part = blockIdx.y, tX = threadIdx.x, tY = threadIdx.y;
    // Tile: the tile of A to operate on
    // Part: which 'part' of ouput this thread calculates towards (refer to report)
    // tX: column of output within tile and part
    // tY: row of output within tile and part
    int c_input = tY*128 + tX, c_kernel = c_input, T_offset = (Tile<<11) + (Part<<10) + c_input, B_offset = (Tile<<14) + c_kernel;
    // the offset of the element of ouput that this thread calculates from input pointer
    // complicated thing: the way that the BtdB kernel lay out its ouput is NHWC BUT we need the input
    // of this kernel (A) in that format so it's almost like we assume A is in NCHW format
    // T_offset: c_input + (enough stuff to skip to wanted tile and part in A)
    // B_offset: c_input + (enought stuff to skip to wanted tile in B since each tile in B has 2^14 elements)
    
    extern __shared__ float input[];
    float *kernel = input + 1024, *out = kernel + 8192;
    // each block gets its shared memory which is divided into 3 parts:
    // input: where one half (becuase each of the 36 tiles get 2 blocks) of input tiles is stored (divided along rows)
    // kernel: where half of the relevant B matrix get stored (divided along columns)
    // out: corespdnding part of output which eventually gets written to C
    int B_stride[32] = {0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048, 2176, 2304, 2432, 2560, 2688, 2816, 2944, 3072, 3200, 3328, 3456, 3584, 3712, 3840, 3968};//, 4096, 4224, 4352, 4480, 4608, 4736, 4864, 4992, 5120, 5248, 5376, 5504, 5632, 5760, 5888, 6016, 6144, 6272, 6400, 6528, 6656, 6784, 6912, 7040, 7168, 7296, 7424, 7552, 7680, 7808, 7936, 8064};
    out[c_input] = 0.0f;

    input[c_input] = A[T_offset];

    for (int k = 0; k < 4; k++) {
        int B_start = B_offset + (k<<12); // 64*64 which is one fourth of 128*128
        kernel[c_kernel] = B[B_start], kernel[c_kernel+1024] = B[B_start+1024];
        kernel[c_kernel+2048] = B[B_start+2048], kernel[c_kernel+3072] = B[B_start+3072];
        __syncthreads();
        // complicated fucking manouver to bring a tile of B into share memory in 4 iterations
        // but it seems like half of the shared memory allocated for kernel is unused

        float sum = 0;
        int y_tmp = (tY<<7)+(k<<5);
        // tY selects row and k goes to the first of the k'th 32 coloumns of that row 
        for (int j = 0; j < 32; j++) {
            sum += input[y_tmp + j] * kernel[tX + B_stride[j]];
            // tX selects column as expected then B_stride[j] goes down j rows
        }
        out[tY*128 + tX] += sum;
        // finally the result of the multiplication of the tY'th row of A and tX'th column of B is calculated
        // but 71 other blocks will add their result to this ()
        __syncthreads();
    }

    C[T_offset] = out[c_input];
}

int kernel_128() {
    float *input_ = get_parameter(inputName128, 16*16*128);
    float *bias = get_parameter(biasName128, 128);
    float *input, *output, *l_weights, *l_bias;
    uint64_t nT1 = 0, nT2 = 0, nT1_cudnn = 0, nT2_cudnn = 0;
    cudaError_t s;

    /////////////////////////////////

    // My Kernel

    /////////////////////////////////


    /*  1. Data preparation  */
    float *t_input, *ip;
    //float *kernel = get_Winograd_Kernel128(weight_winograd_Name128, 128);
    float *kernel = get_parameter(weight_winograd_Name128, 36*128*128);
    float *l_bnBias, *l_bnScale, *bnBias, *bnScale;
    // l: device pointer, bn: batch normalization, kernel: weights, t_input: trasnformed input
    // ip: inner product

    int nInput = 16*16*128, nOutput = 16*16*128, nWeights = 36*128*128, nBias = 128, nTransInput = 16*6*6*128, nInnerProd = 16*6*6*128;
    cudaMalloc((void **) &input, nInput<<3);
    cudaMalloc((void **) &output, nOutput<<2);
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

    
    /*  2. Computing  */
    nT1 = getTimeMicroseconds64();

    kernel_128_winograd_BtdB <<<dim3(4, 4), dim3(128, 6), (6*6*128)<<2 >>> (input, t_input);
    kernel_128_OuterProduct_128<<<dim3(36, 2), dim3(128, 8), (8*128 + 64*128 + 8*128)<<2 >>> (t_input, l_weights, ip);
    kernel_128_winograd_AtIA <<<dim3(4, 4, 128), dim3(6, 6), ((6*6)<<2)>>> (ip, l_bnBias, l_bnScale, output);
    //cudaCheckError();
    cudaDeviceSynchronize();
    
    nT2 = getTimeMicroseconds64();
    printf("TotalTime = %d us\n", nT2-nT1); 


    /*  3. Copy back and free  */
    s = cudaMemcpy(tmp_winograd, output, nOutput<<2, cudaMemcpyDeviceToHost);
    printf("%s\n", cudaGetErrorName(s));
    //cudaCheckError();

    cudaFree(t_input);
    cudaFree(output);
    cudaFree(l_weights);
    cudaFree(l_bias);
    cudaFree(ip);

    free(kernel);
    free(bnScale);
    free(bnBias);


    /////////////////////////////////

    // cuDNN

    /////////////////////////////////

    /*  1. Data preparation  */
    kernel = get_parameter(weight_NCHW_Name128, 9*128*128);
    bnBias = get_parameter(bnBiasName128, 128);
    bnScale = get_parameter(bnScaleName128, 128);
    float* eMean = get_parameter(eMeanName128, 128);
    float* eVar = get_parameter(eVarName128, 128);
    // above two parameters are required for batch norm but in xu's kernel
    // they've been combined into bnBias and bnScale
    float *l_eMean, *l_eVar;
    nInput = 16*16*128, nOutput = 14*14*128, nWeights = 3*3*128*128, nBias = 128;

    cudaMalloc((void **) &output, nOutput<<2);
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


    /*  2. cuDNN preparation  */
    cudnnStatus_t status;
    float one = 1.0, zero = 0.0;
    int size;

    cudnnHandle_t handle;
    status = cudnnCreate(&handle);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed1\n");

    cudnnTensorDescriptor_t xdesc, ydesc, bdesc;
    cudnnFilterDescriptor_t wdesc; // CUDNN_TENSOR_NHWC, CUDNN_TENSOR_NCHW
    status = cudnnCreateTensorDescriptor(&xdesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed2\n");
    status = cudnnSetTensor4dDescriptor(xdesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 16, 16);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed3\n");
    status = cudnnCreateTensorDescriptor(&ydesc);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed4\n");
    status = cudnnSetTensor4dDescriptor(ydesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 1, 128, 14, 14);
    if (status != CUDNN_STATUS_SUCCESS) printf("failed5\n");
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

    status = cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
        &one, &zero, 
        ydesc, output, ydesc, output,
        bnScaleBiasMeanVarDesc, l_bnScale, l_bnBias, l_eMean, l_eVar, CUDNN_BN_MIN_EPSILON);
    // in xu's kernel parts of the batchnorm operation have been taken 'offline'
    if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed2\n");

    status = cudnnActivationForward(handle, act_desc, &one,
        ydesc, output, &zero,
        ydesc, output);
    if (status != CUDNN_STATUS_SUCCESS) printf("Not Successed3\n");

    cudaDeviceSynchronize();
    nT2_cudnn = getTimeMicroseconds64();
    printf("cuDNN TotalTime = %d us\n", nT2_cudnn-nT1_cudnn);


    /*  4. Copy back and free  */
    s = cudaMemcpy(tmp_cudnn, output, nOutput<<2, cudaMemcpyDeviceToHost);
    printf("%s\n", cudaGetErrorName(s));

    cudaFree(extra);
    cudaFree(input);
    cudaFree(output);
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

    output_checker(tmp_winograd, tmp_cudnn, 14, 128, 1);

    return ((nT2-nT1) << 16) | (nT2_cudnn-nT1_cudnn);
    // interesting shifting menouver to return to numbers that are definitly smaller than 2^16
}