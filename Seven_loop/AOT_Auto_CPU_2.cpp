
#include <stdlib.h>
#include "Halide.h"
#include <stdio.h>
#include <iostream>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

using namespace Halide;
using namespace Halide::Internal;

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
	int batch_size = 1;

	Input<Buffer<float,4>> input{"input"};
	Input<Buffer<float,1>> bias{"bias"};
	Input<Buffer<float,4>> weight_1{"weight_1"};
	Input<Buffer<float,4>> weight_2{"weight_2"};
	Output<Buffer<float,4>> output{"output"};

	void generate(){


	//------------------------Algorithm

	//first layer
	Func boundary, conv1, conv2, relu;
	boundary = BoundaryConditions::repeat_edge(input);

    	RDom r(0, size, 0, size, 0, input_channel);
    	conv1(c,x,y,z) = bias(z);
    	conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);

	//second layer
    	RDom R(0, small_size, 0, small_size, 0, output_channel);
    	conv2(c,x,y,z) =  bias(z);
    	conv2(c,x,y,z) += conv1(c ,x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (size*size*output_channel);

	relu(c,x,y,z) = max(0,conv2(c,x,y,z));

	output(c,x,y,z) = relu(c,x,y,z);

	
	
	//------------------------Scheduling

	//autoscheduler just requires the size of buffers.
	if (auto_schedule){
		input.set_estimates({{0,batch_size},{0,width},{0,height},{0,input_channel}});
		bias.set_estimates({{0,output_channel}});
		weight_1.set_estimates({{0,input_channel},{0,size},{0,size},{0,output_channel}});
		weight_2.set_estimates({{0,output_channel},{0,small_size},{0,small_size},{0,output_channel}});
		output.set_estimates({{0,batch_size},{0,width},{0,height},{0,output_channel}});
	}

	else {
	Var xi, xo, yi, yo, ci, co, A, B, C;


	conv1.compute_root()
			.update()
			.vectorize(x,25)
			.parallel(y)
			.parallel(z)
			;
	conv1.print_loop_nest();

	conv2.compute_root()
			.update()
			.vectorize(x,25)
			.parallel(y)
			.parallel(z)
			;

	relu.compute_root();
	}
	} //generate
}; // class


HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer);


