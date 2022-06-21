
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

class Simple_ex : public Generator<Simple_ex>{
	public:
        // Declare some Vars to use below
	Var c, x, y, z;
	int size = 7;
	int small_size = 3;
	int input_channel = 3;
	int output_channel = 64;
	int batch_size = 32;
	int width = 400;
	int height = 600;

	Input<Buffer<float>> input{"input",4};
	Input<Buffer<float>> bias{"bias",1};
	Input<Buffer<float>> weight_1{"weight_1",4};
	Input<Buffer<float>> weight_2{"weight_2",4};
	Output<Buffer<float>> output{"output",4};

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
    	conv2(c,x,y,z) += conv1(c, x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (size*size*output_channel);

	output(c,x,y,z) = max(0,(conv2(c,x,y,z)));

	
	const int vec = natural_vector_size<float>();
	// scheduling

	Var xi, xo, yi, yo, A, B, C;
	

	
	conv1.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		.update()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;
	
	conv2.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		.update()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;

	output.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;
	Target target = find_gpu_target();
	printf("\nGPU Target Hardware : %s\n", target.to_string().c_str());

	} //generate
}; // class


HALIDE_REGISTER_GENERATOR(Simple_ex, simple_ex);



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
        features_to_try.push_back(Target::OpenCL);
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



