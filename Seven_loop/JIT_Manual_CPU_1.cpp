
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
	int width = 400;
	int height = 600;
	int input_channel = 3;
	int output_channel = 64;
    	// Load a grayscale image to use as an input
    	Buffer<float> input(batch_size,width,height,3,"input"); // load_and_convert_image("images/rgb.png");
	Buffer<float> bias(output_channel,"bias");
	//Buffer<float> bias(batch_size,width,height,64,"bias");
    	Buffer<float> weight_1(input_channel, size, size, output_channel, "weight_1");
    	Buffer<float> weight_2(output_channel, size, size, output_channel, "weight_2");
    	const WeightShape wei_1 = {3, 7, 7, 64};
    	const WeightShape wei_2 = {64, 3, 3, 64};
    	//const WeightShape wei_3 = {64, 7, 7, 64};
	Buffer<float> output(batch_size,width,height,64);

	//printf("\nThe input image size is (%d, %d, %d)", input.width(), input.height(), input.channels());
	
	auto start = std::chrono::system_clock::now();
	//Buffer<float> input;
	for (int w = 0; w < batch_size; w++){
	 for (int a = 0; a < width; a++){
            for (int b = 0; b < height ; b++){
            	for (int c = 0; c < input_channel; c++){
	        	input(w,a,b,c) =  (rand() % 10);
		}
	    }
	}
        }
	for (uint8_t a = 0; a < output_channel; a++)
		bias(a) = (rand()%10);

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
	std::cout << latency.count() << " msec \n";

	Func boundary, conv1, conv2, relu1;
	boundary = BoundaryConditions::repeat_edge(input);

	//first layer
    	RDom r(0, size, 0, size, 0, input_channel);
    	conv1(c,x,y,z) = bias(z);//,x,y,z);
    	conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);

	//second layer
    	RDom R(0, small_size, 0, small_size, 0, output_channel);
    	conv2(c,x,y,z) = bias(z);//,x,y,z);
    	conv2(c,x,y,z) += conv1(c, x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z)/(size*size*output_channel);

	relu1(c,x,y,z) = max(0,conv2(c,x,y,z));

	//scheduling
	
	Var xi, xo, yi, yo, ci, co, A, B, C;

	uint8_t Auto = false;

	if (Auto){
		conv1
			// auto opt
			.compute_at(relu1,y)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			.update()
			//.reorder(x, r.x, r.y, z, y, r.z)
			.reorder(c, r.x, x, r.y, y, z, r.z)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			;

	conv2
			// auto opt
			.compute_at(relu1,y)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			.update()
			//.reorder(x, R.x, R.y, z, y, R.z)
			.reorder(c, x, R.x, y, R.y, R.z, z)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			;

	relu1
		.compute_root()
		.reorder(c, z, x, y)
		.split(c,co,ci,8)
		.vectorize(ci)
		.parallel(y)
		;

/*	repeat_edge
		.compute_at(output, x)
		.split(c, co, ci, 8)
		.vectorize(ci)*/

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
	relu1
		.compute_root()
		//.reorder(x,z,y)
		//.split(x,xo,xi,8)
		//.vectorize(xi)
		//.parallel(y)
		;
	}
		

    float avg = 0;	
    int epoch = 2;
    for (unsigned a = 0; a < epoch; a++){

        auto start = std::chrono::system_clock::now();

        Buffer<float> halide_result = relu1.realize({batch_size,width,height,output_channel});  //({input.width(), input.height(), input.channels()});
	auto end = std::chrono::system_clock::now();
	auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	std::cout << latency.count() << " msec \n";
        if (a != 0)	avg += latency.count();
        }
    	//conv2.print_loop_nest();
        if (epoch != 1)	std::cout << "Average latency = " << avg/(epoch-1) << "msec\n";
	
	//relu1.compile_to_llvm_assembly("ll.ll",Target::CUDA,"relu",target);
	
	return 0;
}


