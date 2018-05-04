/*  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
 *  Copyright (C) 2014 Alex Stivala <stivalaa@unimelb.edu.au>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <typeinfo>

#include <stdexcept>

#include <list>
#include <vector>
#include <string>

#ifndef MODEL_HPP_
#define MODEL_HPP_

#define SVN_VERSION_MODEL_HPP "joint activity model rev 7"

/* using consts not enum so it works with Grid */
/* must match values in lib.py */
const int  STRATEGY_COOPERATE = 0;   // cooperator
const int  STRATEGY_DEFECT    = 1;   // defector

typedef enum UpdateRule_e {
  UPDATE_RULE_BESTTAKESOVER = 0,    // best takes over
  UPDATE_RULE_FERMI         = 1,    // Fermi function
  UPDATE_RULE_MODAL         = 2,    // Modal feature (culture only not strategy)
  UPDATE_RULE_MODALBEST     = 3     // Modal feat/strat in equal best payoff agents
} UpdateRule_e;


struct Coord {
   int x, y;
};


// Generic two-dimensional array, supporting a default value.
template<typename T> class Grid {
public:
    Grid(const size_t height, const size_t width, const T def)
		: height(height), width(width) {
    m_data = new T[height*width];
		for (size_t i = 0; i < height; i++) {
			for (size_t j = 0; j < width; j++) {
				m_data[i * width + j] = def;
			}
		}
    }

    ~Grid() {
      delete[] m_data;
    }

    T& operator()(const size_t row, const size_t column) {
        return m_data[row * width +column];
    }

    const T& operator()(const size_t row, const size_t column) const {
        return m_data[row * width + column];
    }

	void print() {
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				std::cout << m_data[i * width +j] << ", ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl;
	}

	void read(const char* file, const char sep) {
		std::ifstream inFile (file);
	    std::string line;
	    int i = 0;
	    while (getline (inFile, line)) {
	        if (i >= height)
	        	std::cout << "too many lines in file" << std::endl;

	        std::istringstream linestream(line);
	        std::string item;
	        int j = 0;
	        while (getline (linestream, item, sep)) {
	            if (j >= width) {
	            	std::cout << "too many columns in line" << std::endl;
	            }
	            m_data[i * width + j] = stringToItem(item);
	            j++;
	        }
	        i++;
	    }

	}



	void write(const char* file, const char sep) {
		std::ofstream outFile(file);

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (j < width - 1)
					outFile << m_data[i * width + j] << sep;
				else
					outFile << m_data[i * width +j];
			}
			if (i < height - 1)
				outFile << std::endl;
		}
	}



protected:
	T stringToItem(const std::string& item);

private:
	int height, width;
  T *m_data;
};


// The following 5 methods have been adopted from here:
// http://www.parashift.com/c++-faq-lite/misc-technical-issues.html

class BadConversion : public std::runtime_error {
	public:
	BadConversion(const std::string& s)
		: std::runtime_error(s) { }
};





template<typename T> inline T convert(const std::string& s,
		bool failIfLeftoverChars = false) {
	T x;
	std::istringstream i(s);
	char c;
	if (!(i >> x) || (failIfLeftoverChars && i.get(c) && i.gcount() > 0))
		throw BadConversion(s);
	return x;
}





template <> int Grid<int>::stringToItem(const std::string& item) {
	return convert<int>(item);
}


template <> double Grid<double>::stringToItem(const std::string& item) {
    return convert<double>(item);
}



template<typename T> inline std::string toString(const T& x) {
  std::ostringstream o;
  if (!(o << x))
    throw BadConversion(std::string("stringify(")
                        + typeid(x).name() + ")");
  return o.str();
}






double similarity(Grid<int>& C, const int F, const int a, const int b);


#endif /* MODEL_HPP_ */
