all: main
	./main

main: main.cu src/environment.cpp include/config.hpp include/environment.hpp
	nvcc -arch=sm_75 -std=c++17 -O3 -Iinclude main.cu src/environment.cpp -o main

environment_tests: tests/environment_tests.cpp src/environment.cpp include/config.hpp include/environment.hpp
	g++ -std=c++17 -O2 -Iinclude tests/environment_tests.cpp src/environment.cpp -o environment_tests

.PHONY: all main tests
tests: environment_tests
	./environment_tests
