# Alex Stivala
# January 2013

import random
import numpy

def ef(c0, q, veclist, power2, p1, p2):
    """
    Neutral evolution from an initial vector to 2^power2 vectors as rows in matrix
   
    Arguments:
      c0 - initial integer vector
      q  - possible range of each element (0:(q-1))
      veclist - initial list (usually [] for caller)
      power2 - 2^power2 rows will be generated
      p1 - probability of a mutation in one element at each step (left subtree)
      p2 probability of a mutation in one element at each step (right subtr

    Return value:
      list of 2^power2 vectors, each vector is terminal in evolved tree
    """
    if power2 == 0:
        returnveclist = list(veclist)                 
        returnveclist.append(c0) # awkward, but list.append is not an operator
        return returnveclist
    else:
        c1 = numpy.array(c0)
        c2 = numpy.array(c0)
        if random.random() < p1:
            i = random.randint(0, len(c0)-1)
            c1[i] = random.randint(0, q-1)
        if random.random() < p2:
            j = random.randint(0, len(c0)-1)
            c2[j] = random.randint(0, q-1)
        return (ef(c1, q, veclist, power2-1, p1, p2) + 
                ef(c2, q, veclist, power2-1, p1, p2))



def perturb(m, q, p):
    """
    Change each element in the matrix m to a random value in 0:(q-1)
    with probability p

    Parameteters:
       m - numpy array to perturnb
       q - range of integers possible values
       p - probability of chaning each element in m
    Return value:
       perturbed matrix
    """
    return numpy.vectorize(lambda x : x if random.random() >= p else random.randint(0, q-1)) (m)

def ef2(c0, q, veclist, power2, k):      
    """
    Neutral evolution from an initial vector to 2^power2 vectors as rows in matrix
   
    Arguments:
      c0 - initial integer vector
      q  - possible range of each element (0:(q-1))
      veclist - initial list (usually [] for caller)
      power2 - 2^power2 rows will be generated
      k - max number of elements to randomly chagne each step

    Return value:
      list of 2^power2 vectors, each vector is terminal in evolved tree
    """
    if power2 == 0:
        returnveclist = list(veclist)                 
        returnveclist.append(c0) # awkward, but list.append is not an operator
        return returnveclist
    else:
        c1 = numpy.array(c0)
        c2 = numpy.array(c0)
        for s in range(k):
            i = random.randint(0, len(c0)-1)
            c1[i] = random.randint(0, q-1)
        for s in range(k):
            j = random.randint(0, len(c0)-1)
            c2[j] = random.randint(0, q-1)
        return (ef2(c1, q, veclist, power2-1, k) + 
                ef2(c2, q, veclist, power2-1, k))



def prototype_evolve(F, q, n, k, t):
    """
    Create set of culture vectors based on k initial prototypes,
    by radomly changing small enough fraction of traits so that new
    vectors are close to a (randomly chosen) prototype vector
   
    Arguments:
       F - vector dimension
       q - number of values a trait can take (integer 0..q-1)
       n - number of vectors to create (total including k prototypes)
       k - number of prototypes
       t - max number of traits to mutate frmo prototype for new vector
   
     Return value:
       list of n vectors with k prototypes as first k vectors
    """
    assert(t < F)
    # create k prototypes as first k rows
    veclist = [numpy.array([random.randrange(q) for i in range(F)]) for j in range(k)]

    # create the rest of the vectors by choosing a prototype and making
    # a new vector with up to t of its traits mutated
    for i in xrange(n-k):
        prototype = veclist[random.randrange(k)]
        cnew = numpy.array(prototype)
        for s in xrange(t):
            j = random.randrange(F)
            cnew[j] = random.randrange(q)
        veclist.append(cnew)
    return veclist


def trivial_ultrametric(F, q, n):
    """
    Create set of culture vectors that are perfectly ultrametric,
    by generating orthognal vectors (and their multiples)
    of dimension F, so that it generates (q-1)*F vectors, and 
    then randomly sample n of these (without replacement).
    Note that therefore must have n <= (q-1)*F
   
    They are perfectly ultrametric (wrt Hamming distance)
    but it is something of a trivial
    case in that each vector only has 2 traits different between
    each other vector (so Hamming distance is always 2 between any
    pair of vectors)

    Arguments:
       F - vector dimension
       q - number of values a trait can take (integer 0..q-1)
       n - number of vectors to create 
   
     Return value:
       list of n vectors that satisfy ultrametric inequality

    """
    assert(n <= (q-1)*F)
    veclist = []
    for i in xrange(1,q):
        for j in xrange(F):
            v = numpy.array(F*[0])
            v[j] = i
            veclist.append(v)
    return random.sample(veclist, n)
