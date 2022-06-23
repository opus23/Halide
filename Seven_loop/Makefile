BASIC=~/halide_build/distrib
LIB=$(BASIC)/lib
INC=$(BASIC)/include
TOOLS=$(BASIC)/tools

HALIDE_DIR=~/halide_build/distrib
HALIDE_LIB=$(HALIDE_DIR)/lib
HALIDE_TOOL=$(HALIDE_DIR)/tools
HALIDE_INC=$(HALIDE_DIR)/include



1:JIT_Manual_CPU_1.cpp
	g++ JIT_Manual_CPU_1.cpp -o bin/conv -I ~/halide_build/distrib/include -I ~/halide_build/distrib/tools -L ~/halide_build/distrib/lib -std=c++17 -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -O3
	./bin/conv


2: AOT_Auto_CPU_2.cpp
	g++ AOT_Auto_CPU_2.cpp $(TOOLS)/GenGen.cpp -g -std=c++17 -fno-rtti -I $(INC) -L $(LIB) -lHalide -lpthread -ldl -o bin/conv_generate
	export LD_LIBRARY_PATH=$(LIB)   # For linux
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_false -e static_library,h,schedule target=host auto_schedule=false
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_true -e static_library,h,schedule -p $(LIB)/libautoschedule_mullapudi2016.so -s Mullapudi2016 target=host auto_schedule=true machine_params=32,16777216,40
	g++ AOT_Auto_CPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_false.a bin/auto_schedule_true.a -ldl -lpthread -o bin/AOT_CPU
	./bin/AOT_CPU


3: JIT_Manual_GPU_3.cpp
	g++ JIT_Manual_GPU_3.cpp -g -std=c++17 -I ~/halide_build/distrib/include -I ~/halide_build/distrib/tools -L ~/halide_build/distrib/lib -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o bin/conv -O3
	./bin/conv


4: AOT_Auto_GPU_4.cpp
	g++ AOT_Auto_GPU_4.cpp $(TOOLS)/GenGen.cpp -g -std=c++17 -fno-rtti -I $(INC) -L $(LIB) -lHalide -lpthread -ldl -o bin/conv_generate
	export LD_LIBRARY_PATH=$(LIB)   # For linux	
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_false -e static_library,h,schedule target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41 auto_schedule=false
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_true -e static_library,h,schedule -p $(LIB)/libautoschedule_li2018.so -s Li2018 target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41-cuda_capability_75 auto_schedule=true machine_params=32,16777216,40
	g++ AOT_Auto_GPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_true.a bin/auto_schedule_false.a  -ldl -lpthread -o bin/AOT_GPU
	./bin/AOT_GPU



clean:
	rm -rf bin/*
