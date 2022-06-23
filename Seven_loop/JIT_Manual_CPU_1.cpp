
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
	int batch_size = 1;
	int width = 256;
	int height = 512;
	int input_channel = 3;
	int output_channel = 64;
    	// Load a grayscale image to use as an input
   	Buffer<float> input(batch_size,width,height,3,"input");
	Buffer<float> bias(output_channel,"bias");
   	Buffer<float> weight_1(input_channel, size, size, output_channel, "weight_1");
   	Buffer<float> weight_2(output_channel, size, size, output_channel, "weight_2");
   	const WeightShape wei_1 = {3, 7, 7, 64};
   	const WeightShape wei_2 = {64, 3, 3, 64};
	Buffer<float> output(batch_size,width,height,64);

	
	auto start = std::chrono::system_clock::now();
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
					weight_1(a,b,c,d) = (rand() % 10);
                   }
			}
		}
    }

	for (int a = 0; a < output_channel; a++){
        for (int b = 0; b < small_size; b++){
          	for (int c = 0; c < small_size; c++){
          	    for (int d = 0; d < output_channel; d++){
			    	weight_2(a,b,c,d) = (rand() % 10);
                }
			}
	    }
    }

	auto end = std::chrono::system_clock::now();
	auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	std::cout << latency.count() << " msec \n";



	//-----------------------------Algorithm

	Func boundary, conv1, conv2, relu1;
	boundary = BoundaryConditions::repeat_edge(input);

	//first layer
    	RDom r(0, size, 0, size, 0, input_channel);
    	conv1(c,x,y,z) = bias(z);
    	conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);

	//second layer
    	RDom R(0, small_size, 0, small_size, 0, output_channel);
    	conv2(c,x,y,z) = bias(z);//,x,y,z);
    	conv2(c,x,y,z) += conv1(c, x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z)/(size*size*output_channel);

	relu1(c,x,y,z) = max(0,conv2(c,x,y,z));



	//----------------------------Scheduling
	
	Var xi, xo, yi, yo, ci, co, A, B, C;

	uint8_t Auto = false;

	if (Auto){
	//Important : This schedule is pre-tuned by AOT_Auto_CPU_2.cpp(autoscheduler).
	//			  This file cannot produce auto-tuned schedule.
	conv1
			.compute_at(relu1,y)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			.update()
			.reorder(c, r.x, x, r.y, y, z, r.z)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			;

	conv2
			.compute_at(relu1,y)
			.split(x, xo, xi, 8)
			.vectorize(xi)
			.update()
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


	}
	else{
		conv1
			//Hand-tuned by Hyunjun
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
			.compute_root()
			.update()
			.vectorize(x,25)
			//.split(y,yo,yi,2)	
			//.fuse(yi, x, B)
			.parallel(y)
			.parallel(z)
			//.unroll(R.x)
			//.unroll(R.y)
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
    int epoch = 5;
    for (unsigned a = 0; a < epoch; a++){

        auto start = std::chrono::system_clock::now();

        Buffer<float> halide_result = relu1.realize({batch_size,width,height,output_channel});
		auto end = std::chrono::system_clock::now();
		auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
		std::cout << latency.count() << " msec \n";
        if (a != 0)	avg += latency.count();
    }
    //relu.print_loop_nest();
    if (epoch != 1)	std::cout << "Average latency = " << avg/(epoch-1) << "msec\n";
	
	return 0;
}


