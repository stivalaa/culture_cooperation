CXXFLAGS = -Wall -Wextra -O3 --fast-math -fno-strict-aliasing -std=c++0x
CXXFLAGS_DEBUG = -Wall -Wextra -O0 -pg --fast-math -fno-strict-aliasing -g -std=c++0x

model: model.o 
	g++ model.o -o model
    
model.o: src/model.cpp 
	g++ $(CXXFLAGS) -c src/model.cpp


debug:
	g++ $(CXXFLAGS_DEBUG) -c src/model.cpp
	g++ $(CXXFLAGS_DEBUG) model.o -o model

clean:
	rm model model.o 
