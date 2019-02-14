CC=clang++
FLAGS=-std=c++11 -O3
INCUDES=-I/Library/Frameworks/Ultrahaptics.framework/Headers
LIBS=-lm -framework Ultrahaptics

HEADERS=CyclicUltrahapticsArray.hpp

%.o: %.cpp $(HEADERS)
	$(CC) -c -o $@ $< $(FLAGS) $(INCUDES)

array_control: CyclicUltrahapticsArray.o TCPArray.o array_control.o
	$(CC) -o $@ $^ $(FLAGS) $(INCLUDES) $(LIBS)

.PHONY: clean
clean:
	rm -f *.gch
	rm -f *.o