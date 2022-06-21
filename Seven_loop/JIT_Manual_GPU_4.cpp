
#include <stdio.h>
#include "Halide.h"
#include "clock.h"
#include "halide_image_io.h"
#include <chrono>
#include <iostream>
#include <stdlib.h>

using namespace Halide;
using namespace Halide::Tools;

Target find_gpu_target();

// Define some Vars to use.
Var A, c, x, y, z, i, ii, xo, yo, xi, yi;

class MyPipeline {
public:
    Func conv1, conv2, relu1, conv1_cast, conv2_cast, boundary;
    Buffer<float> input;
    Buffer<float> weight_1;
    Buffer<float> weight_2;
    Buffer<float> bias;


    MyPipeline(Buffer<float> in, Buffer<float> W1, Buffer<float> W2, Buffer<float> B1)
        : input(in), weight_1(W1), weight_2(W2), bias(B1) {
	//Var x, y, z;
	float size = 7;
	float small_size = 3;
	float input_channel = 3;
	float output_channel = 64;
	float width = 400;
	float height = 600;
	float batch_size = 32;

	boundary = BoundaryConditions::repeat_edge(input);
        RDom r(0, size, 0, size, 0, input_channel);
        conv1(c,x,y,z) = bias(z);
        conv1(c,x,y,z) += boundary(c, x+r.x, y+r.y, r.z) * weight_1(r.z, r.x, r.y, z) / (size*size*input_channel);
        conv1_cast(c,x,y,z) = cast<float>(conv1(c,x,y,z));

        RDom R(0, small_size, 0, small_size, 0, output_channel);
	conv2(c,x,y,z) = bias(z);
        conv2(c,x,y,z) += conv1_cast(c, x+R.x, y+R.y, R.z) * weight_2(R.z, R.x, R.y, z) / (small_size*small_size*output_channel);
    	
	conv2_cast(c,x,y,z) = cast<float>(conv2(c,x,y,z));

	relu1(c,x,y,z) = max(conv2_cast(c,x,y,z), 0);
    }


    void schedule_for_cpu() {
	relu1.compute_root()
		//.reorder(x,z,y)
		//.split(x,xo,xi,8)
		//.vectorize(xi)
		//.parallel(y)
		//.update()
		//.reorder(z,y,x)
		//.bound(z,0,64)
		//.unroll(z)
		;

	conv2.compute_root()
		.update()
		.vectorize(x,25)
		.parallel(y)
		.parallel(z)
		;
	
	conv1.compute_root()
		.update()
		.vectorize(x,25)
		.parallel(y)
		.parallel(z)
		;
	/*conv1.compute_at(relu1,y)
		.split(x, xo, xi, 8)
		.vectorize(xi)
		.update()
		.reorder(x, r.x, r.y, z, y, r.z)
		.split(x, xo, xi, 8)
		.vectorize(xi)
		;

	conv2.compute_at(relu1,y)
		.split(x, xo, xi, 8)
		.vectorize(xi)
		.update()
		.reorder(x, R.x, R.y, z, y, R.z)
		.split(x, xo, xi, 8)
		.vectorize(xi)
		;*/

        Target target = get_host_target();
        printf("\nCPU Target hareware : %s\n\n", target.to_string().c_str());
	relu1.compile_jit(target);
	//relu1.print_loop_nest();
    }


    // Now a schedule that uses CUDA or OpenCL.
    bool schedule_for_gpu() {
        Target target = find_gpu_target();
        printf("\nGPU Target hareware : %s\n\n", target.to_string().c_str());
        if (!target.has_gpu_feature()) {
            return false;
        }

	relu1.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;

	conv2.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		.update()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;
	
	conv1.compute_root()
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		.update()
		.reorder(x,y,z,c)
		.gpu_tile(x, y, xo, yo, xi, yi, 16,8)
		;
	
	relu1.compile_jit(target);
	relu1.print_loop_nest();

        return true;
    }

    void test_performance() {
        Buffer<float> output(32,400,600,64);
        relu1.realize(output);
        double avg_time = 0.0;
        for (int i = 0; i < 5; i++) {
            double t1 = current_time();
            // Run the filter 100 times.
            //for (int j = 0; j < 1; j++) {
                relu1.realize(output);
            //}
            // Force any GPU code to finish by copying the buffer back to the CPU.
            output.copy_to_host();
            double t2 = current_time();
            double elapsed = (t2 - t1);
	    if (i != 0)	avg_time += elapsed;
            //if (i == 0 || elapsed < best_time)       best_time = elapsed;
        }
	avg_time = avg_time / 4;
        printf("%1.4f milliseconds\n", avg_time);
    }
};

int main(int argc, char **argv) {
	float size = 7;
	float small_size = 3;
	float input_channel = 3;
	float output_channel = 64;
	float width = 400;
	float height = 600;
	float batch_size = 32;
    Target target = get_host_target();
    printf("\nTarget hareware : %s\n\n", target.to_string().c_str());

    	double t9 = current_time();
    Buffer<float> input(batch_size,width,height,input_channel,"input");
    for (int c = 0; c < batch_size; c++){
      for (int x = 0; x < width; x++){
        for (int y = 0; y < height; y++){
          for (int z = 0; z < input_channel; z++){
		  input(c,x,y,z) = rand() % 10;
          }
        }
      }
    }
    Buffer<float> weight_1(input_channel,size,size,output_channel,"weight_1");
    for (int c = 0; c < input_channel; c++){
      for (int x = 0; x < size; x++){
        for (int y = 0; y < size; y++){
          for (int z = 0; z < output_channel; z++){
		  weight_1(c,x,y,z) = rand() % 10;
          }
        }
      }
    }
    Buffer<float> weight_2(output_channel,small_size,small_size,output_channel,"weight_2");
    for (int c = 0; c < output_channel; c++){
      for (int x = 0; x < small_size; x++){
        for (int y = 0; y < small_size; y++){
          for (int z = 0; z < output_channel; z++){
		  weight_2(c,x,y,z) = rand() % 10;
          }
        }
      }
    }
    Buffer<float> bias(64,"bias");
          for (int z = 0; z < 64; z++){
		  bias(z) = rand() % 10;
          }
    	double t10 = current_time();
    	printf("%1.4f milliseconds load buffer\n", t10-t9);

    // Allocated an image that will store the correct output
    //Buffer<float> reference_output(400,600,64);

    printf("Running pipeline on CPU:\n");
	double t1 = current_time();
    MyPipeline p1(input, weight_1, weight_2, bias);
    p1.schedule_for_cpu();
    	double t2 = current_time();
    	printf("%1.4f milliseconds CPU pipeline\n", t2-t1);
	
	double t3 = current_time();
    //p1.relu1.realize(reference_output);
    
    //auto start = std::chrono::system_clock::now();
    //Buffer<float> reference_output = p1.relu1.realize({32,400,600,64});
    //auto end = std::chrono::system_clock::now();
    //auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    //std::cout << latency.count() << " msec is evaluated by chrono \n";
    	double t4 = current_time();
    	printf("%1.4f milliseconds CPU realize latency\n", t4-t3);

    printf("Running pipeline on GPU:\n");
    MyPipeline p2(input, weight_1, weight_2, bias);
    	double t7 = current_time();
    bool has_gpu_target = p2.schedule_for_gpu();
    	double t8 = current_time();
    	printf("%1.4f milliseconds GPU pipeline\n", t8-t7);
    if (has_gpu_target) {
        //printf("Testing GPU correctness:\n");
        //p2.test_correctness(reference_output);
    } else {
        printf("No GPU target available on the host\n");
    }

    //printf("Testing performance on CPU:\n");
    //p1.test_performance();

    if (has_gpu_target) {
        printf("Testing performance on GPU:\n");
        p2.test_performance();
    }

    return 0;
}

// A helper function to check if OpenCL, Metal or D3D12 is present on the host machine.

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













/*
void test_correctness(Buffer<float> reference_output) {
        Buffer<float> output =
            relu1.realize({input.width(), input.height(), input.channels()});

        // Check against the reference output.
        for (int c = 0; c < input.channels(); c++) {
            for (int y = 0; y < input.height(); y++) {
                for (int x = 0; x < input.width(); x++) {
                    if (output(x, y, c) != reference_output(x, y, c)) {
                        printf("Mismatch between output (%d) and "
                               "reference output (%d) at %d, %d, %d\n",
                               output(x, y, c),
                               reference_output(x, y, c),
                               x, y, c);
                        exit(-1);
                    }
                }
            }
        }
    }
};
*/
