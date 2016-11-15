CC=gcc
CFLAGS=-I

barry: sobel.o
	$(CC) -o barry sobel.o $(CFLAGS)