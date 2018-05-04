"""
 hackmodel.py - model.py not provided in source code from
 
 http://ww2.cs.mu.oz.au/~pfauj/physicaa2012/physicaa2012-cpp.zip

 but model.similarity and model.distance are called so I wrote these
 python implementatinos assuming they do the same as their corresponding
 functions in physicaa2012-cpp/src/model.cpp

Alex Stivala
Mon Oct  8 15:39:48 EST 2012
"""

import numpy
from numpy import linalg
from math import sqrt

def similarity(A, B):
    """
    Calculate the cultural similarity between culture vectors A and B given the
    proportion of features they have in common.

    Parameters:
       A - numpy vector of culutural traits for agent A
       B - numpy vecto ro cultural traits for agent B

    Return value:
      similarity between A and B 
    """
    F = numpy.size(A)
    return float(numpy.count_nonzero(A == B)) / float( F)


def distance(A, B, m, toroidal):
    """
    Calculate the Eucliden distance on the lattice between agents A and B

    Parameters:
       A - (x,y) location of agent A
       B - (x,y) location of agent B
       m - unused
       toroidal -must be False

    Return value
       Euclidean distacne between A and B
       Actual return tuple with this twice since callers uses [1] on it,
       no idea what it is really meant for
    """
# linalg.norm will work with any number of dimensions, not just 2
#    d = linalg.norm(numpy.array(A) - numpy.array(B))
    d = sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
    return (d,d)


def manhattan_distance(A, B, toroidal=False):
    """
    Calculate the Manhattan distance on the lattice between agents A and B

    Parameters:
       A - (x,y) location of agent A
       B - (x,y) location of agent B
       toroidal -must be False

    Return value
       Manhattan distacne between A and B
    """
    assert(not toroidal) # toroidal not supported
    d = abs(A[0] - B[0]) + abs(A[1] - B[1])
    return d


