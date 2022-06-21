
#include <stdlib.h>
#include "Halide.h"
#include <stdio.h>
#include <iostream>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

using namespace Halide;
//#include "halide_image_io.h"
//using namespace Halide::Tools;
using namespace Halide::Internal;

Target find_gpu_target();

class ConvolutionLayer : public Generator<ConvolutionLayer>{
	public:
        // Declare some Vars to use below
	Var c, x, y, z;
	int width = 256;
	int height = 512;
	int size = 7;
	int small_size = 3;
	int input_channel = 3;
	int output_channel = 64;
	int batch_size = 32;

	Input<Buffer<float,4>> input{"input"};
	Input<Buffer<float,1>> bias{"bias"};
	Input<Buffer<float,4>> weight_1{"weight_1"};
	Input<Buffer<float,4>> weight_2{"weight_2"};
	Output<Buffer<float,4>> output{"output"};

	void generate(){

	//first layer
	Func boundary, conv1, conv2;
	boundary = BoundaryConditions::repeat_edge(input);

    	RDom r(0, size, 0, size, 0, input_channel);
    	conv1(c,x,y,z) = bias(z);
    	conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);

	//second layer
    	RDom R(0, small_size, 0, small_size, 0, output_channel);
    	conv2(c,x,y,z) =  bias(z);
    	conv2(c,x,y,z) += conv1(c ,x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (small_size*small_size*output_channel);

	output(c,x,y,z) = max(0,conv2(c,x,y,z));

	
	//const int vec = natural_vector_size<float>();
	// scheduling

	if (auto_schedule){
	Target target = find_gpu_target();
	printf("\nauto Target Hardware : %s\n", target.to_string().c_str());
		input.set_estimates({{0,batch_size},{0,width},{0,height},{0,input_channel}});
		bias.set_estimates({{0,output_channel}});
		weight_1.set_estimates({{0,input_channel},{0,size},{0,size},{0,output_channel}});
		weight_2.set_estimates({{0,output_channel},{0,small_size},{0,small_size},{0,output_channel}});
		output.set_estimates({{0,batch_size},{0,width},{0,height},{0,output_channel}});

	}

	else {
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

	output.compute_root()
		//.gpu_tile(x,y,xo,yo,xi,yi,4,8)
		.split(x,xi,xo,64, Halide::TailStrategy::ShiftInwards)
		.fuse(c,xi,c)
		//.fuse(c,y,c)
		.fuse(c,z,c)
		.reorder(xo,c)
		.reorder_storage(x,c,y,z)
		.gpu_blocks(c)
		.gpu_threads(xo)
		;

	output.print_loop_nest();

	Target target = find_gpu_target();
	printf("\nGPU Target Hardware : %s\n", target.to_string().c_str());


	}
	} //generate
}; // class


HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer);


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
        features_to_try.push_back(Target::CUDA);
        //features_to_try.push_back(Target::OpenCL);
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



