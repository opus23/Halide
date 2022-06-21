
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

class ConvolutionLayer : public Generator<ConvolutionLayer>{
	public:
        // Declare some Vars to use below
	Var c, x, y, z;
	int width = 400;
	int height = 600;
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
    	conv2(c,x,y,z) += conv1(c ,x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (size*size*output_channel);

	output(c,x,y,z) = max(0,conv2(c,x,y,z));

	
	//const int vec = natural_vector_size<float>();
	// scheduling

	if (auto_schedule){
		input.set_estimates({{0,batch_size},{0,width},{0,height},{0,input_channel}});
		bias.set_estimates({{0,output_channel}});
		//bias.set_estimates({{0,batch_size},{0,width},{0,height},{0,output_channel}});
		weight_1.set_estimates({{0,input_channel},{0,size},{0,size},{0,output_channel}});
		weight_2.set_estimates({{0,output_channel},{0,small_size},{0,small_size},{0,output_channel}});
		output.set_estimates({{0,batch_size},{0,width},{0,height},{0,output_channel}});

	}

	else {
	Var xi, xo, yi, yo, A, B, C;
	conv1.compute_root()
			//.reorder(z, y, x, c)
			.update()
			.vectorize(x,25)
			//.split(y,yo,yi,2)	
			//.fuse(yi, x, A)
			.parallel(y)
			.parallel(z)
			//.unroll(r.x)
			//.unroll(r.y)
			;

	conv2.compute_root()
			//.reorder(z, y, x, c)
			.update()
			.vectorize(x,25)
			//.split(y,yo,yi,2)	
			//.fuse(yi, x, B)
			.parallel(y)
			.parallel(z)
			//.unroll(R.x)
			//.unroll(R.y)
			;

	output.compute_root();

	}
	} //generate
}; // class


HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv_layer);


