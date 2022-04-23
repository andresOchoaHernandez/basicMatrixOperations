#!/bin/bash

g++ -std=c++11 ./src/BasicMatrixOperations.cpp ./main.cpp -I./include -o app
./app
rm app