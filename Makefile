all:
	nvcc -arch=sm_75 -O3 -std=c++17 main.cu -lcublas -o main && ./main