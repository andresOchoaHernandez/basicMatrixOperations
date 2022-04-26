flags     := -std=c++11 -Wall

inc       := include/
src       := src/BasicMatrixOperations.cpp

test      := test/test.cpp 
test_exec := bin/tests

main      := main.cpp
exec      := bin/app

all: tests main

tests:
	@g++ $(src) $(test) -I$(inc) -o$(test_exec)
	@./$(test_exec)
	@rm $(test_exec) 
main:   
	@g++ $(src) $(main) -I$(inc) -o$(exec)
	@./$(exec) 
clean:
	@rm $(exec)