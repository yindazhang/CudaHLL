all: cuhll.bin cuhll-debug.bin cuhll-profile.bin

SOURCE_FILE=cuhll.cu

# optimized binary
cuhll.bin: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# debug binary without optimizations
cuhll-debug.bin: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# optimized binary with line number information for profiling
cuhll-profile.bin: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_75,code=[compute_75,sm_75] $^ -lcublas -o $@

# NB: make sure you change the --algo flag here to profile the one you care about. 
# You can change the --export flag to set the filename of the profiling report that is produced.
profile: cuhll-profile.bin
	sudo /usr/local/cuda-11.8/bin/ncu --export my-profile --set full ./cuhll-profile.bin --size=1024 --reps=1 --algo=1 --validate=false

clean:
	rm -f cuhll*.bin
