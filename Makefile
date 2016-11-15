CC=g++
CFLAGS=-I

barry: sobel.o
	$(CC) -msse4 -O2 -openmp -fopenmp -o barry sobel.cpp $(CFLAGS) `pkg-config opencv --cflags --libs`
