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


3: AOT_Manual_CPU_3.cpp 
	g++ -g $(HALIDE_DIR)/tools/GenGen.cpp AOT_Manual_CPU_3.cpp -g -fno-rtti -I $(HALIDE_TOOL) -I $(HALIDE_INC) -L ${HALIDE_LIB} -lHalide -std=c++17  -o bin/simple_generator  -lz -lpthread -ldl -std=c++17
	./bin/simple_generator -g simple_ex -o bin -e bitcode,c_source,llvm_assembly,static_library,registration,stmt,assembly target=x86-64-linux-avx2-enable_llvm_loop_opt
	g++ ~/halide_build/distrib/tools/RunGenMain.cpp bin/simple_ex.a bin/simple_ex.registration.cpp -I ~/halide_build/distrib/include -o bin/AOT_CPU -lpng -ljpeg -lz -lpthread -ldl -std=c++17
	./bin/AOT_CPU input=random:0:[4,400,600,3] bias=random:0:[64] weight_1=random:0:[3,7,7,64] weight_2=random:0:[64,3,3,64] --benchmarks=all --benchmark_min_time=1 --parsable_output

#33: AOT_Manual_CPU_3.cpp
#	g++ -g $(HALIDE_DIR)/tools/GenGen.cpp AOT_Manual_CPU_3.cpp -g -fno-rtti -I $(HALIDE_TOOL) -I $(HALIDE_INC) -L ${HALIDE_LIB} -lHalide -std=c++17  -o bin/simple_generator  -lz -lpthread -ldl -std=c++17
#	./bin/simple_generator -g simple_ex -o bin -e bitcode,c_source,llvm_assembly,static_library,registration,stmt,assembly target=x86-64-linux-avx2-enable_llvm_loop_opt
#	
#	g++ ~/halide_build/distrib/tools/RunGenMain.cpp bin/simple_ex.a bin/simple_ex.registration.cpp -I ~/halide_build/distrib/include -o bin/AOT_CPU -lpng -ljpeg -lz -lpthread -ldl -std=c++17
#	./bin/AOT_CPU input=random:0:[4,400,600,3] bias=random:0:[4,400,600,64] weight_1=random:0:[3,7,7,64] weight_2=random:0:[64,7,7,64] --benchmarks=all --benchmark_min_time=1 --parsable_output




4: JIT_Manual_GPU_4.cpp
	g++ JIT_Manual_GPU_4.cpp -g -std=c++17 -I ~/halide_build/distrib/include -I ~/halide_build/distrib/tools -L ~/halide_build/distrib/lib -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -o bin/conv -O3
	#target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41-cuda_capability_75 
	./bin/conv


5: JIT_Manual_GPU_5.cpp
	g++ JIT_Manual_GPU_5.cpp -o bin/conv -I ~/halide_build/distrib/include -I ~/halide_build/distrib/tools -L ~/halide_build/distrib/lib -std=c++17 -lHalide `libpng-config --cflags --ldflags` -ljpeg -lpthread -ldl -O3
	./bin/conv


6: AOT_Manual_GPU_6.cpp
	g++ -g $(HALIDE_DIR)/tools/GenGen.cpp AOT_Manual_GPU_6.cpp -g -fno-rtti -I $(HALIDE_TOOL) -I $(HALIDE_INC) -L ${HALIDE_LIB} -lHalide -std=c++17  -o bin/simple_generator  -lz -lpthread -ldl -std=c++17
	./bin/simple_generator -g simple_ex -o bin -e bitcode,c_source,llvm_assembly,static_library,registration,stmt,assembly target=x86-64-linux-avx-avx2-f16c-fma-opencl-sse41
	g++ ~/halide_build/distrib/tools/RunGenMain.cpp bin/simple_ex.a bin/simple_ex.registration.cpp -I ~/halide_build/distrib/include -o bin/AOT_GPU -lpng -ljpeg -lz -lpthread -ldl -std=c++17
	./bin/AOT_GPU input=random:0:[32,400,600,3] bias=random:0:[64] weight_1=random:0:[3,7,7,64] weight_2=random:0:[64,3,3,64] --benchmarks=all --benchmark_min_time=1 --parsable_output
	

7: AOT_Auto_GPU_7.cpp
	g++ AOT_Auto_GPU_7.cpp $(TOOLS)/GenGen.cpp -g -std=c++17 -fno-rtti -I $(INC) -L $(LIB) -lHalide -lpthread -ldl -o bin/conv_generate
	export LD_LIBRARY_PATH=$(LIB)   # For linux	
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_false -e static_library,h,schedule target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41 auto_schedule=false
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_true -e static_library,h,schedule -p $(LIB)/libautoschedule_li2018.so -s Li2018 target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41-cuda_capability_75 auto_schedule=true #machine_params=32,16777216,40
	g++ AOT_Auto_GPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_true.a bin/auto_schedule_false.a  -ldl -lpthread -o bin/AOT_GPU
	./bin/AOT_GPU



	#./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_false -e static_library,h,schedule target=x86-64-linux-avx-avx2-f16c-fma-opencl-sse41 auto_schedule=false
	#./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_true -e static_library,h,schedule -p $(LIB)/libautoschedule_li2018.so -s Li2018 target=x86-64-linux-avx-avx2-f16c-fma-cuda-sse41-opencl auto_schedule=true machine_params=32,16777216,40
	
	#g++ AOT_Auto_GPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_false.a  -ldl -lpthread -o bin/AOT_GPU



llvm: AOT_Auto_GPU_7.cpp
	clang++ -c -emit-llvm AOT_Auto_GPU_7.cpp $(TOOLS)/GenGen.cpp -g -std=c++17 -fno-rtti -I $(INC) -L $(LIB) -lHalide -lpthread -ldl -o bin/conv_generate.bc
	export LD_LIBRARY_PATH=$(LIB)   # For linux
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_false -e static_library,h,schedule target=x86-64-linux-avx-avx2-f16c-fma-opencl-sse41 auto_schedule=false
	./bin/conv_generate -o bin/ -g conv_layer -f auto_schedule_true -e static_library,h,schedule -p $(LIB)/libautoschedule_mullapudi2016.so -s Mullapudi2016 target=x86-64-linux-avx-avx2-f16c-fma-opencl-sse41 auto_schedule=true machine_params=32,16777216,40
	#g++ AOT_Auto_GPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_false.a  -ldl -lpthread -o bin/AOT_GPU
	g++ AOT_Auto_GPU_run.cpp -std=c++17 -I $(INC) -I $(TOOLS) bin/auto_schedule_true.a bin/auto_schedule_false.a  -ldl -lpthread -o bin/AOT_GPU
	./bin/AOT_GPU


clean:
	rm -rf bin/*
