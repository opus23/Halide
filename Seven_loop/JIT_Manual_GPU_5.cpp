
#include <stdlib.h>
#include "Halide.h"
#include <stdio.h>
#include <iostream>
#include <chrono>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#include "clock.h"
using namespace Halide;
#include "halide_image_io.h"
using namespace Halide::Tools;
using namespace Halide::Internal;

Target find_gpu_target();

struct WeightShape{
	int c;
	int w;
	int h;
	int n;
};



int main(int argc, char **argv) {


        // Declare some Vars to use below
	Var c("c"), x("x"), y("y"), z("z");
	int size = 7;
	int small_size = 3;
	int batch_size = 32;
	int width = 256;
	int height = 512;
	int input_channel = 3;
	int output_channel = 64;
    	// Load a grayscale image to use as an input
    	Buffer<float> input(batch_size,width,height,input_channel,"input"); // load_and_convert_image("images/rgb.png");
	Buffer<float> bias(output_channel,"bias");
	//Buffer<float> bias(batch_size,width,height,output_channel,"bias");
    	Buffer<float> weight_1(input_channel, size, size, output_channel, "weight_1");
    	Buffer<float> weight_2(output_channel, small_size, small_size, output_channel, "weight_2");
    	const WeightShape wei_1 = {3, 7, 7, 64};
    	const WeightShape wei_2 = {64, 3, 3, 64};
    	//const WeightShape wei_3 = {64, 7, 7, 64};
	//Buffer<float> output(batch_size,width,height,output_channel);

	//printf("\nThe input image size is (%d, %d, %d)", input.width(), input.height(), input.channels());
	
	auto start = std::chrono::system_clock::now();
	//Buffer<float> input;
	for (int w = 0; w < batch_size; w++){
	 for (int a = 0; a < width; a++){
            for (int b = 0; b < height ; b++){
            	for (int c = 0; c < 3; c++){
	        	input(w,a,b,c) =  (rand() % 10);
		}
	    }
	}
        }
	//for (int w = 0; w < batch_size; w++){
	// for (int a = 0; a < width; a++){
        //    for (int b = 0; b < height ; b++){
            	for (int c = 0; c < 64; c++){
	        	bias(c) =  (rand() % 10);
		}
	//    }
	// }
	//}

	for (int a = 0; a < input_channel; a++){
            for (int b = 0; b < size; b++){
            	for (int c = 0; c < size; c++){
            	    for (int d = 0; d < output_channel; d++){
		//weight is 1 for identification
	        	weight_1(a,b,c,d) = (rand() % 10);
                    }
		}
	    }
        }

	for (int a = 0; a < output_channel; a++){
            for (int b = 0; b < small_size; b++){
            	for (int c = 0; c < small_size; c++){
            	    for (int d = 0; d < output_channel; d++){
		//weight is 1 for identification
	        	weight_2(a,b,c,d) = (rand() % 10);
                    }
		}
	    }
        }

	auto end = std::chrono::system_clock::now();
	auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	std::cout << "data load : " << latency.count() << " msec \n";

	Func boundary, conv1, conv2, relu;
	boundary = BoundaryConditions::repeat_edge(input);

	//first layer
    	RDom r(0, size, 0, size, 0, input_channel);
    	conv1(c,x,y,z) = bias(z);
    	conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);

	//second layer
    	RDom R(0, small_size, 0, small_size, 0, output_channel);
    	conv2(c,x,y,z) = bias(z);
    	conv2(c,x,y,z) += conv1(c, x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (size*size*output_channel);

	relu(c,x,y,z) = max(0,conv2(c,x,y,z));

	//scheduling
	
	Var xi, xo, yi, yo, A, B, C;

	uint8_t Use_GPU = true;

	if (Use_GPU){

	Var xi, xo, yi, yo, A, B, C, xii, xoo;

	boundary.compute_root()
		.split(_1,xi,xo,64, Halide::TailStrategy::ShiftInwards)
		.fuse(_0,xi,_0)
		//.fuse(_0,_2,_0)
		.fuse(_0,_3,_0)
		.reorder(xo,_0)
		.reorder_storage(_1,_0,_2,_3)
		.gpu_blocks(_0)
		.gpu_threads(xo)
		;

	conv1.compute_root()
		.split(x,xi,xo,64, Halide::TailStrategy::ShiftInwards)
		.fuse(c,xi,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xo,c)
		.reorder_storage(x,c,y,z)
		.gpu_blocks(c)
		.gpu_threads(xo)
		.update()
		.split(x,xii,xoo,64, Halide::TailStrategy::GuardWithIf)
		.fuse(c,xii,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xoo,c)
		.gpu_blocks(c)
		.gpu_threads(xoo)
			;

	conv2.compute_root()
		.split(x,xi,xo,64, Halide::TailStrategy::ShiftInwards)
		.fuse(c,xi,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xo,c)
		.reorder_storage(x,c,y,z)
		.gpu_blocks(c)
		.gpu_threads(xo)
		.update()
		
		.split(x,xii,xoo,64, Halide::TailStrategy::GuardWithIf)
		.fuse(c,xii,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xoo,c)
		.gpu_blocks(c)
		.gpu_threads(xoo)
		;

	relu.compute_root()
		.split(x,xi,xo,64, Halide::TailStrategy::ShiftInwards)
		.fuse(c,xi,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xo,c)
		.reorder_storage(x,c,y,z)
		.gpu_blocks(c)
		.gpu_threads(xo)
		;


/*		conv1.compute_root()
			.gpu_tile(x,y,xo,yo,xi,yi,16,8)
			.update()
			.gpu_tile(x,y,xo,yo,xi,yi,16,8)
			;

		conv2.compute_root()
			.gpu_tile(x,y,xo,yo,xi,yi,16,8)
			.update()
			.gpu_tile(x,y,xo,yo,xi,yi,16,8)
			;

		relu.compute_root()
			.gpu_tile(x,y,xo,yo,xi,yi,16,8)
			;*/


		Target target = find_gpu_target();
		//Target cuda_target = "x86-64-linux-avs-avx2-f16c-fma-cuda-sse41";
		printf("\nGPU Target Hardware : %s\n", target.to_string().c_str());
		relu.compile_jit(target);

		//std::vector<Argument> args;
		//args.push_back(arg);
		//relu.compile_to_llvm_assembly("ll.ll",args,"relu",target);
	}
	else{
		conv1
			//Human opt
			.compute_root()
			.update()
			.vectorize(x,25)
			//.split(y,yo,yi,2)	
			//.fuse(yi, x, A)
			.parallel(y)
			.parallel(z)
			//.unroll(r.x)
			//.unroll(r.y)
			;
		
		conv2	
			// Human opt
			.compute_root()
			.update()
			.vectorize(x,25)
			//.split(y,yo,yi,2)	
			//.fuse(yi, x, B)
			.parallel(y)
			.parallel(z)
			//.unroll(R.x)
			//.unroll(R.y)
			
			//.tile(x,y,xi,yi,40,60)
			//.reorder(z,y,x)
			//.split(x,xo,xi,25)
			//.unroll(xi)
			;
	relu
		.compute_root()
		//.reorder(x,z,y)
		//.split(x,xo,xi,8)
		//.vectorize(xi)
		//.parallel(y)
		;
	}
		

    float avg = 0;	
    int epoch = 5;
    Buffer<float> output(32,256,512,64);


    for (unsigned a = 0; a < epoch; a++){

        //auto start = std::chrono::system_clock::now();
        double t1 = current_time();
	//Buffer<float> halide_result = relu.realize(output);  //({input.width(), input.height(), input.channels()});
	relu.realize(output);  //({input.width(), input.height(), input.channels()});
	output.copy_to_host();
	std::cout << "Result : " << output(1,1,1,1) << "\n";
	//auto end = std::chrono::system_clock::now();
	//auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
        double t2 = current_time();
	double elapsed = (t2-t1);
	printf("%1.4f millisecond\n", elapsed);
	//d::cout << latency.count() << " msec \n";
        if (a != 0)	avg += elapsed;
        }
    	conv2.print_loop_nest();
        if (epoch != 1)	std::cout << "Average latency = " << avg/(epoch-1) << "msec\n";
	return 0;
}




Target find_gpu_target() {
    // Start with a target suitable for the machine you're running this on.
    Target target = get_host_target();

    std::vector<Target::Feature> features_to_try;
    if (target.os == Target::Windows) {
        // Try D3D12 first; if that fails, try OpenCL.
        if (sizeof(void*) == 8) {
            // D3D12Compute support is only available on 64-bit systems at present.
            features_to_try.push_back(Target::D3D12Compute);
        }
        features_to_try.push_back(Target::OpenCL);
    } else if (target.os == Target::OSX) {
        // OS X doesn't update its OpenCL drivers, so they tend to be broken.
        // CUDA would also be a fine choice on machines with NVidia GPUs.
        features_to_try.push_back(Target::Metal);
    } else {
        features_to_try.push_back(Target::CUDA);//CUDA
        //features_to_try.push_back(Target::OpenCL);//CUDA
    }
    // Uncomment the following lines to also try CUDA:
    // features_to_try.push_back(Target::CUDA);

    for (Target::Feature f : features_to_try) {
        Target new_target = target.with_feature(f);
        if (host_supports_target_device(new_target)) {
            return new_target;
        }
    }

    printf("Requested GPU(s) are not supported. (Do you have the proper hardware and/or driver installed?)\n");
    return target;
}



