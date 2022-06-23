// Halide tutorial lesson 21: Auto-Scheduler

// Before reading this file, see lesson_21_auto_scheduler_generate.cpp

// This is the code that actually uses the Halide pipeline we've
// compiled. It does not depend on libHalide, so we won't be including
// Halide.h.
//
// Instead, it depends on the header files that lesson_21_auto_scheduler_generator produced.
#include "bin/auto_schedule_false.h"
#include "bin/auto_schedule_true.h"

// We'll use the Halide::Runtime::Buffer class for passing data into and out of
// the pipeline.
#include "HalideBuffer.h"
#include "halide_benchmark.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int batch_size = 1;
    int input_channel = 3; 
    int output_channel = 64;
    int width = 256;
    int height = 512;
    int size = 7;
    int small_size = 3; 

    // Let's declare and initialize the input images
    Halide::Runtime::Buffer<float> input(batch_size,width,height,input_channel);
    Halide::Runtime::Buffer<float> bias(output_channel);
    //Halide::Runtime::Buffer<float> bias(batch_size,width,height,output_channel);
    Halide::Runtime::Buffer<float> weight_1(input_channel,size,7,output_channel);
    Halide::Runtime::Buffer<float> weight_2(output_channel,small_size,small_size,output_channel);
    Halide::Runtime::Buffer<float> output(batch_size,width, height,output_channel);
    
    for (int c = 0; c < batch_size; c++) {
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
		for (int z = 0; z < input_channel; z++){
                    input(c, x, y, z) = rand();
		}
            }
        }
    }
    //for (int c = 0; c < output_channel; c++) {
    //    for (int x = 0; x < height; x++) {
    //        for (int y = 0; y < width; y++) {
		for (int z = 0; z < output_channel; z++){
                    bias(z) = rand();
		}
  //          }
    //    }
    //}

    for (int c = 0; c < output_channel; c++) {
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
            	for (int x = 0; x < input_channel; x++) {
                    weight_1(x, y, z, c) = rand();
                }
	    }
	}
    }


    for (int c = 0; c < output_channel; c++) {
        for (int z = 0; z < small_size; z++) {
            for (int y = 0; y < small_size; y++) {
            	for (int x = 0; x < output_channel; x++) {
                    weight_2(x, y, z, c) = rand();
                }
	    }
	}
    }

    // Run each version of the codes (with no auto-schedule and with
    // auto-schedule) multiple times for benchmarking.
    double auto_schedule_off = Halide::Tools::benchmark(1, 2, [&]() {
        auto_schedule_false(input, bias, weight_1, weight_2, output);
    });
    printf("Manual schedule: %gms\n", auto_schedule_off * 1e3);

    double auto_schedule_on = Halide::Tools::benchmark(1, 2, [&]() {
        auto_schedule_true(input, bias, weight_1, weight_2, output);
    });
    printf("Auto schedule: %gms\n", auto_schedule_on * 1e3);

    // auto_schedule_on should be faster since in the auto_schedule_off version,
    // the schedule is very simple.
/*    if (!(auto_schedule_on < auto_schedule_off)) {
        fprintf(stderr, "Warning: expected auto_schedule_on < auto_schedule_off , "
                        "saw auto_schedule_on=%f auto_schedule_off=%f\n", auto_schedule_on, auto_schedule_off); \
    }
*/
    return 0;
}
