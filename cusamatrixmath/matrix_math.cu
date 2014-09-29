#include "matrix_math.h"
#define len 50 //max = 100000000
#define LOOPS 10
int main (int argc, char** argv){
	float *a = new float[len]; 
	float *result = new float[len]; 
	for(long i= 0; i < len; i++){
		a[i] = 1.0;
	}
	//////////////////////////////////////////////////////
	///////////////// initialize Cuda ///////////////////
	////////////////////////////////////////////////////
	float *dev_a, *dev_tmpA, *dev_tmpB, *dev_result, *dev_seg;
	cudaMalloc( (void**)&dev_a,       len    * sizeof(float) );
	cudaMalloc( (void**)&dev_tmpA,    len    * sizeof(float) );
	cudaMalloc( (void**)&dev_tmpB,    len    * sizeof(float) );
	cudaMalloc( (void**)&dev_result,  len    * sizeof(float) );
	cudaMemcpy( dev_a, a, len * sizeof(float), cudaMemcpyHostToDevice );



	clock_t t;
	///////////////////////////////////////////////////////
	///////////////   sequential scan   //////////////////
	/////////////////////////////////////////////////////
	
	t = clock(); //start timer
	for(int i = 0; i < LOOPS; i++){
		seq_scan(a, result, len);
	}
	t = clock() - t; //end timer
	printf ("Sequential Scan took %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	cout << result[len - 1] << "\n";
	//print_array(result, len);
	
	////////////////////////////////////////////////////////////////
	/////////////////    NAIVE SCAN ///////////////////////////////
	//////////////////////////////////////////////////////////////
	
	t = clock();
	for(int i = 0; i < LOOPS; i++){
		naive_scan(dev_a,dev_tmpA,dev_tmpB,dev_result, len);
	}
	t = clock() - t;
	printf ("Naive GPU took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	cudaMemcpy( result, dev_result, len * sizeof(float), cudaMemcpyDeviceToHost );
	//print_array(result, len);
	cout << result[len - 1] << "\n";
	

	////////////////////////////////////////////////////////////////
	/////////////////   SHARED SCAN ///////////////////////////////
	//////////////////////////////////////////////////////////////
	
	t = clock();
	for(int i = 0; i < LOOPS; i++){
		shared_scan(dev_a, dev_result, len);
	}
	t = clock() - t;
	printf ("Shared GPU took me %d clicks (%f seconds).\n",t,((float)t)/CLOCKS_PER_SEC);
	cudaMemcpy( result, dev_result, len * sizeof(float), cudaMemcpyDeviceToHost );
	//print_array(result, len);
	cout << result[len - 1] << "\n";
	
	////////////////////////////////////////////////////////////////
	/////////////////   String Compaction /////////////////////////
	//////////////////////////////////////////////////////////////
	float* sparse_array = new float[10];
	for(int i = 0; i<10; i += 2){
		sparse_array[i] = 0.0;
		sparse_array[i + 1] = i + 1;
	}
	print_array(sparse_array, 10);
	float* compact_array = new float[10];
	string_compaction(sparse_array, compact_array, 10);
	print_array(compact_array, 10);


	cudaFree(dev_a);
	cudaFree(dev_tmpA);
	cudaFree(dev_tmpB);
	cudaFree(dev_result);
	free(a);
	free(result);
	free(sparse_array);
	free(compact_array);
	std::cin.get();
}

void seq_scan(float* arr, float* result, int length){
	result[0] = arr[0];
	for (int i = 1; i < length; i++){
		result[i] = result[i-1] + arr[i];
	}
}

void seq_scatter(float* in_arr, float* scan_arr,  float* result, int length){
	if(scan_arr[0] == 1){
		result[0] = in_arr[0];
	}
	for(int i = 1; i < length; i++){
		if (scan_arr[i] > scan_arr[i - 1]){
			result[(int)scan_arr[i] - 1] = in_arr[i];
		}
	}
}

void naive_scan( float *dev_a, float *dev_tmpA, float* dev_tmpB, float *dev_result, int length){
	int threads = min(32, length);
	int block = ceil((float)length / (float)threads);
	int segCount = ceil((float)length / (float)threads);
	
	if(length == threads){
		glob_excl_block_scan<<< 1,threads>>>( dev_a, dev_tmpA, dev_result, length); ///////////
		return;
	}
	float *dev_seg;
	cudaMalloc( (void**)&dev_seg,    segCount  * sizeof(float) );
	
	glob_sec_scan<<< block,threads>>>( dev_a, dev_tmpA,dev_tmpB, dev_result, dev_seg, length); //Segmented scan

	//scan resulting segments
	naive_scan(dev_seg, dev_tmpA, dev_tmpB, dev_seg, block);
	/*
	float* segs = new float[block];
	cudaMemcpy( segs, dev_seg, block * sizeof(float), cudaMemcpyDeviceToHost );
	print_array(segs, block);
	*/
	add_segments<<< block, threads >>>( dev_result, dev_seg, length);
	cudaFree(dev_seg);
}

void shared_scan( float *dev_a, float *dev_result, int length){
	int threads = min(32, length);
	int block = ceil((float)length / (float)threads);
	int segCount = ceil((float)length / (float)threads);
	
	if(length == threads){
		//glob_excl_block_scan<<< 1,threads>>>( dev_a, dev_tmpA, dev_result, length); //use shared mem
		shared_block_scan<<< 1, threads, 2 * threads * sizeof(float) >>>( dev_a, dev_result, length);
		return;
	}
	float *dev_seg;
	cudaMalloc( (void**)&dev_seg,    segCount  * sizeof(float) );
	
	shared_naive_scan<<< block,threads, threads * 2 * sizeof(float)>>>( dev_a, dev_result, dev_seg, length);//Segmented scan
	
	
	//scan resulting segments
	shared_scan(dev_seg, dev_seg, block);
	/*
	float* segs = new float[block];
	cudaMemcpy( segs, dev_seg, block * sizeof(float), cudaMemcpyDeviceToHost );
	print_array(segs, block);
	*/
	add_segments<<< block, threads >>>( dev_result, dev_seg, length);
	cudaFree(dev_seg);
}

void string_compaction( float* in_array, float* comp_array, int length){
	int threads = min(32, length);
	int block = ceil((float)length / (float)threads);
	
	float* debug = new float[10];

	float *dev_a, *dev_result, *dev_compact, *dev_bool;
	cudaMalloc( (void**)&dev_a,       length    * sizeof(float) );
	cudaMalloc( (void**)&dev_bool,       length    * sizeof(float) );
	cudaMalloc( (void**)&dev_result,  length    * sizeof(float) );
	cudaMalloc( (void**)&dev_compact, length    * sizeof(float) );
	cudaMemcpy( dev_a, in_array, length * sizeof(float), cudaMemcpyHostToDevice );

	make_bool_arr<<< block, threads >>>(dev_a, dev_bool, length);
	shared_scan(dev_bool, dev_result, length);
	cudaMemcpy( debug, dev_result, length * sizeof(float), cudaMemcpyDeviceToHost );
	print_array(debug, 10);
	
	scatter<<< block, threads>>>(dev_a, dev_result, dev_compact, length);
	
	
	cudaMemcpy( comp_array, dev_compact, length * sizeof(float), cudaMemcpyDeviceToHost );
}

__global__ void make_bool_arr( float* in_arr, float* result, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( in_arr[index] > 0){
		result[index] = 1.0;
	}else{
		result[index] = 0.0;
	}
}

__global__ void scatter( float* in_arr, float* scan_arr, float* result, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[index] = 0;
	__syncthreads();
	if( index > length || index == 0){
		if(scan_arr[0] == 1.0){
			result[0] = in_arr[0];
		}
		return;
	}
	//not first element and not off end of array
	if(scan_arr[index] > scan_arr[index - 1]){
		result[(int)scan_arr[index] - 1] =  in_arr[index];
	}
}

__global__ void shared_block_scan( float *a, float *result, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ float shared[]; // twice the size of block
	if(index < length){
		shared[threadIdx.x] = a[index]; //result[index] = shared[idx.x]

		__syncthreads(); //to be sure result array is fully initialized
		int imt = index % blockDim.x;
		for(int d = 1; d < blockDim.x; d*=2){
			if(imt  >= d){
				shared[blockDim.x + threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x - d];
			}else{
				shared[blockDim.x + threadIdx.x] = shared[threadIdx.x];
			}
			//can just switch the pointers
			__syncthreads(); //ensures tmp array is fully updated
			shared[threadIdx.x] = shared[blockDim.x + threadIdx.x]; //update results for next iteration
			__syncthreads;  //ensures result array is fully updated
		}
		result[index] = shared[blockDim.x + threadIdx.x];
	}
}

__global__ void glob_excl_block_scan( float *a, float *tmp, float *result, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < length){
		result[index] = a[index];
		__syncthreads(); //to be sure result array is fully initialized
		int imt = index % blockDim.x;
		for(int d = 1; d < blockDim.x; d*=2){
			if(imt  >= d){
				tmp[index] = result[index] + result[index - d];
			}else{
				tmp[index] = result[index];
			}
			//can just switch the pointers
			__syncthreads(); //ensures tmp array is fully updated
			result[index] = tmp[index]; //update results for next iteration
			__syncthreads;  //ensures result array is fully updated
		}
	}
}

__global__ void glob_sec_scan( float *a, float *tmpA, float* tmpB, float *result, float *segment, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < length){
		tmpB[index] = a[index];
		__syncthreads(); //to be sure result array is fully initialized
		int imt = index % blockDim.x;
		for(int d = 1; d < blockDim.x; d*=2){
			if(imt  >= d){
				tmpA[index] = tmpB[index] + tmpB[index - d];
			}else{
				tmpA[index] = tmpB[index];
			}
			//can just switch the pointers
			__syncthreads(); //ensures tmp array is fully updated
			tmpB[index] = tmpA[index]; //update results for next iteration
			__syncthreads;  //ensures result array is fully updated
		}
		result[index] = tmpB[index];
		segment[blockIdx.x] = result[(blockIdx.x * blockDim.x) + blockDim.x -1];
	}
}

__global__ void add_segments( float *result, float *segment, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < length){
		int seg = index / blockDim.x - 1; 
		if(seg >= 0){
			result[index] += segment[seg];
		}
	}
}

__global__ void shared_naive_scan( float *a, float *result, float *segment, int length){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ float shared[]; // twice the size of block
	if(index < length){
		shared[threadIdx.x] = a[index]; //result[index] = shared[idx.x]

		__syncthreads(); //to be sure result array is fully initialized
		int imt = index % blockDim.x;
		for(int d = 1; d < blockDim.x; d*=2){
			if(imt  >= d){
				shared[blockDim.x + threadIdx.x] = shared[threadIdx.x] + shared[threadIdx.x - d];
			}else{
				shared[blockDim.x + threadIdx.x] = shared[threadIdx.x];
			}
			//can just switch the pointers
			__syncthreads(); //ensures tmp array is fully updated
			shared[threadIdx.x] = shared[blockDim.x + threadIdx.x]; //update results for next iteration
			__syncthreads;  //ensures result array is fully updated
		}
		result[index] = shared[blockDim.x + threadIdx.x];
		segment[blockIdx.x] = shared[blockDim.x - 1]; //result[(blockIdx.x * blockDim.x) + blockDim.x -1];
	}
}

void print_array(float* arr, int length){
	for(int i = 0; i < length; i++){
		std::cout << arr[i] << ",";
	}
	std::cout << "\n";
}