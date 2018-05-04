
#  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
#  Copyright (C) 2015 Alex Stivala <stivalaa@unimelb.edu.au>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

# modified by ADS to use MPI and seed srand with time+pid and to also 
# ensure unique tmp files so safe for parallel execution with MPI Python
# NB this involved extra command line pameter to model to not compatible
# with unmodified versions
#
# This version is also simplified to remove stuff to do with social networks
# etc. for use with simpler Axelrod type models on lattice to not waste
# time/memory on these computations. Note the columns are left in output
# however to maintain compatibility with shell scripts etc. from more
# complex versions.

import os, re, glob, sys
from random import randrange, random
from igraph import Graph, ADJ_UNDIRECTED
from numpy import array, nonzero, zeros
from stats import ss, lpearsonr, mean
from collections import deque
import numpy as np



import time

#import model
import hackmodel

import csv
import commands
import ConfigParser


# must match strategy constants in model.hpp
class STRATEGY:
    COOPERATE = 0
    DEFECT    = 1


# must match UpdateRule_e enum in model.hpp
class UPDATE_RULE:
    UPDATE_RULE_BESTTAKES_OVER = 0
    UPDATE_RULE_FERMI          = 1
    UPDATE_RULE_MODAL          = 2
    UPDATE_RULE_MODALBEST      = 3


class INIT_STRATEGY:
    RANDOM =     0    # uniform random
    STRIPE =     1    # cooperate on one side, defect the other
    CHESSBOARD = 2    # checkerboard pattern
    
# Randomly draw one element from s based on the distribution given in w.
def randsample(s,w):
    cum = 0.0
    randTmp = random() * sum(w)
    
    for i in range(len(w)):
        cum += w[i]
        if randTmp <= cum:
            return s[i]



def unique_rows(a):
    """
    Given numpy 2d array a, return numpy 2d array with only the unique
    rows in a
    From:
    http://stackoverflow.com/questions/8560440/removing-duplicate-columns-and-rows-from-a-numpy-2d-array/8567929#8567929
    """
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))




# Returns a culture-dependent membership vector of the nodes in vector c and
# the number of cultures found overall among agents in c.
def cultures_in(C,c):
    # Label all nodes with their culture, whereby agents with the same culture
    # get the same label
    vertex_label = list([0]*len(c))
    
    cultures = list()
    
    vertex_label[0] = 0
    cultures.append(C[c[0]])
    current_label = 1
    for i in range(1,len(c)):        
        prev = -1
        for j in range(i):
            if len(nonzero(C[c[i]]-C[c[j]])[0]) == 0:
                prev = j
                break
        if prev >= 0:
            vertex_label[i] = vertex_label[prev]
        else:
            vertex_label[i] = current_label
            cultures.append(C[c[i]])
            current_label = current_label + 1
    
    
    # The number of cultures can be determined by the current pointer to the
    # label to be assigned next
    return vertex_label, current_label, cultures

# Calculates the diversity within those agents' cultures in C that are given in
# the list cluster.
def calcDiversityOfCluster(C, cluster):
    n = len(cluster)
    
    diversity = 0.0
    for i in range(n):
        for j in range(i):
            diversity += 1.0 - hackmodel.similarity(C[cluster[i]], C[cluster[j]])
    
    if n == 1:
        return 0.0
    else:
        return diversity / (n*(n-1)/2)


# Calculates the modal/prototype culture among the group of agents in C that is
# determined by the list cluster.
def calcModal(C, cluster, F, q):
    M = zeros((F, q))
    
    modal = array([0] * F)
    
    for i in range(len(cluster)):
        for j in range(F):
            M[j, C[cluster[i]][j]] += 1
    
    for i in range(F):
        max = 0
        maxJ = 0
        for j in range(q):
            if M[i,j] > max:
                maxJ = j
                max = M[i,j]
            modal[i] = maxJ
            
    return modal


# Calculates the within-cluster and between-cluster diversity as well as
# the within-/-between-cluster diversity ratio and the mapping of cluster size
# to diversity of the clusters given in the list clusters, whose cultures are
# defined in C.
def calcDiversity(C, clusters, F, q):
    wcd = 0.0
    modals = list()
    size_x_div = list()
   
    for i in range(len(clusters)):
        div = calcDiversityOfCluster(C, clusters[i])
        wcd += div
        modals.append(calcModal(C, clusters[i], F, q))
        size_x_div.append([len(clusters[i]), div])
        
    wcd = wcd / len(clusters)
    bcd = calcDiversityOfCluster(modals, range(len(modals)))
    
    if wcd == 0.0 or wcd == float('nan'):
        diversity_ratio = float('nan')
    else:
        diversity_ratio = bcd / wcd
    
    return wcd, bcd, diversity_ratio, size_x_div
    

# Calculates the Pearson correlation between two list, left and right.
def calcCorrelation(network, left, right):
    if network and ss(left) != 0 and ss(right) != 0 and max(left) != 0 and max(right) != 0:
        return lpearsonr(left,right)[0]
    else:
        return float('nan')    

def mooreNeighbourhood(x, y, m):
    """
    Return the list of agents in the Moore neighbourhood (radius 1, so 8 
    positions) of the given (x,y) position on lattice
    
    Parameters:
      x  - x co-ordinate 
      y  - y -co-ordinate
      m  - lattice dimension, x and y are from 0..m-1

    Return value:
      List of (up to 8) tuples (x, y) of positions on lattice
    """
    neighbour_list = []
    for xi in [-1, 0, 1]:
        for yi in [-1, 0, 1]:
            # don't include (x,y) itself, and Chebyshev distance <= 1
            if (xi != 0 or yi != 0) and max(abs(xi), abs(yi)) <= 1:
                bx = x + xi
                by = y + yi
                # handle edge of lattice, not toroidal
                if bx >= 0 and bx < m and by >= 0 and by < m:
                    neighbour_list.append((bx, by))

    assert(len(neighbour_list) >= 3 and len(neighbour_list) <= 8)
    return neighbour_list

def vonNeumannNeighbourhood(x, y, m):
    """
    Return the list of agents in the von Neumann neighbourhood (radius 1, so 4 
    positions) of the given (x,y) position on lattice
    
    Parameters:
      x  - x co-ordinate 
      y  - y -co-ordinate
      m  - lattice dimension, x and y are from 0..m-1

    Return value:
      List of (up to 8) tuples (x, y) of positions on lattice
    """
    neighbour_list = []
    for xi in [-1, 0, 1]:
        for yi in [-1, 0, 1]:
            # don't include (x,y) itself, and Manhattan distance <= 1
            if (xi != 0 or yi != 0) and abs(xi) + abs(yi) <= 1:
                bx = x + xi
                by = y + yi
                # handle edge of lattice, not toroidal
                if bx >= 0 and bx < m and by >= 0 and by < m:
                    neighbour_list.append((bx, by))

    assert(len(neighbour_list) >= 2 and len(neighbour_list) <= 4)
    return neighbour_list


def growCulturalRegions(m, n, C, L):
    """
    Find the cultural regions (specifically the number of regions
    and the size of the largest regions) using a region growing algorihtm.

    Parameters:
       m - lattice dimension, x and y coords are 0..m-1
       n - number of agents
       C - cultures, list of culture vectors (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent
   
    Return value:
       tuple (numRegions, maxRegionSize): number of cultural regions and
       max region size.
    """
    # build reverse dictionary of L; Lreverse[(x,y)] is agent id at (x,y)
    Lreverse = dict([(coord, agent) for (agent, coord) in enumerate(L)])
    maxRegionSize = 0
    thisRegionSize = 0
    notvisited = set(range(n))  # set of agents not processed yet
    region = dict()             # dict mappnig agent to region number
    current_region = 0          # current region number
    queue = deque()             # use as queue, append() and popleft()
    
    while len(notvisited) > 0 :
        seed = notvisited.pop()  # random seed agent
        queue.append(seed)
        region[seed] = current_region
        thisRegionSize = 1
        while len(queue) > 0:
            a = queue.popleft()
            # NB using von Neumann neighbourhood here for use
            # with models such as
            # lattice-jointactivity-simcoop-social-noise-cpp-end/model
            # where the von Neumann neighbourhood is used, but note some
            # other models such as
            # lattice-schelling-meanoverlap-network/model
            # used Moore neighbourhood
            for coords in vonNeumannNeighbourhood(L[a][0], L[a][1], m): # 4-connected
                b = Lreverse[coords]
                assert(a != b)
                # if b not processed yet and has same culture vector as a
                if ( (not region.has_key(b)) and
                     hackmodel.similarity(C[a], C[b]) == 1 ):
                    queue.append(b)
                    region[b] = current_region
                    notvisited.remove(b)
                    thisRegionSize += 1
                if thisRegionSize > maxRegionSize:
                    maxRegionSize = thisRegionSize
        current_region += 1
    numRegions = current_region
    return (numRegions, maxRegionSize)
    

def calcNumAndMaxSizeOfLatticeCulturalRegions(m, n, C, L):
    """
    Compute the number of regions and the largest region size, where
    region here is the traditional Axelrod region on the lattice,
    i.e. agents with the same culture in contiguous sites on the 
    lattice (a 'cultural region' in Axelrod's terminology)

    Parameters:
       m - lattice dimension, x and y coords are 0..m-1    
       n - number of agents
       C - cultures, list of culture vectors (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent

    Return value:
       tuple (numRegions, maxRegionSize): number of cultural regions and
       maximum size of a cultural region.
       Both are normlized by dividing by n.
    """
    assert(len(C) == n)
    assert(len(L) == n)
    # we can conventiently compute these by building a 'cultural region graph'
    # which is the graph where two nodes (agents) are connected exactly
    # when both (a) they have the same culture and (b) they are adjacent
    # on the lattice (their Manhattan distance is 1).
    #
    # NB using von Neumann neighbourhood (Manhattan distance = 1)
    # here for use with models such as
    # lattice-jointactivity-simcoop-social-noise-cpp-end/model where
    # the von Neumann neighbourhood is used, but note some other models
    # such as lattice-schelling-meanoverlap-network/model used Moore
    # neighbourhood
    
    # first = time.time()    
    # cultural_region_graph = Graph(n)
    # cultural_region_graph.add_edges(
    #         [(i, j) for i in xrange(n) for j in xrange(i)
    #             if hackmodel.similarity(C[i], C[j]) == 1 and
    #                hackmodel.manhattan_distance(L[i], L[j]) == 1 ] 
    # )
    # components = cultural_region_graph.components()
    # print 'graph cultural regions: ', time.time() - first
    
    # but building the graph is too slow for larger lattices, so
    # use region growing algorithm (no recursion) instead
#    first = time.time()
    (num_regions, max_region_size) = growCulturalRegions(m, n, C, L)
#    print 'nonrecursive region growing cultural regions: ', time.time() - first

#    print 'xxx', 'num_regions =',num_regions, ' components  =',len(components)
#    print 'yyy', 'max_region_size =', max_region_size, ' max compopnent size = ',max(components.sizes())
    # assert(num_regions == len(components))
    # assert(max_region_size == max(components.sizes()))

    normalized_num_regions = float(num_regions) / n
    normalized_max_region_size = float(max_region_size) / n
    return (normalized_num_regions, normalized_max_region_size)


def growStrategyRegions(m, n, Strategy, L):
    """
    Find the strategy regions (specifically the number of regions
    and the size of the largest regions) using a region growing algorihtm.

    Parameters:
       m - lattice dimension, x and y coords are 0..m-1
       n - number of agents
       Strategy - strategies, list of strategy integers (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent
   
    Return value:
       tuple (numRegions, maxRegionSize): number of strategy regions and
       max region size.
    """
    # build reverse dictionary of L; Lreverse[(x,y)] is agent id at (x,y)
    Lreverse = dict([(coord, agent) for (agent, coord) in enumerate(L)])
    maxRegionSize = 0
    thisRegionSize = 0
    notvisited = set(range(n))  # set of agents not processed yet
    region = dict()             # dict mappnig agent to region number
    current_region = 0          # current region number
    queue = deque()             # use as queue, append() and popleft()
    
    while len(notvisited) > 0 :
        seed = notvisited.pop()  # random seed agent
        queue.append(seed)
        region[seed] = current_region
        thisRegionSize = 1
        while len(queue) > 0:
            a = queue.popleft()
            # NB using von Neumann neighbourhood here for use
            # with models such as
            # lattice-jointactivity-simcoop-social-noise-cpp-end/model
            # where the von Neumann neighbourhood is used, but note some
            # other models such as
            # lattice-schelling-meanoverlap-network/model
            # used Moore neighbourhood
            for coords in vonNeumannNeighbourhood(L[a][0], L[a][1], m): # 4-connected
                b = Lreverse[coords]
                assert(a != b)
                # if b not processed yet and has same strategy as a
                if ( (not region.has_key(b)) and
                     Strategy[a] == Strategy[b] ):
                    queue.append(b)
                    region[b] = current_region
                    notvisited.remove(b)
                    thisRegionSize += 1
                if thisRegionSize > maxRegionSize:
                    maxRegionSize = thisRegionSize
        current_region += 1
    numRegions = current_region
    return (numRegions, maxRegionSize)


def calcNumAndMaxSizeOfLatticeStrategyRegions(m, n, Strategy, L):
    """
    Compute the number of regions and the largest region size, where
    region here is the strategy region on the lattice,
    i.e. agents with the same strategy=thisStat in contiguous sites on the 
    lattice

    Parameters:
       m - lattice dimension, x and y coords are 0..m-1    
       n - number of agents
       Strategy - strategies, list of strategy integers (one for each agent)
       L - lattice, list of (x,y) tuples, one for each agent

    Return value:
       tuple (numRegions, maxRegionSize): number of strategy regions and
       maximum size of a strategy region.
       Both are normlized by dividing by n.
    """
    assert(len(Strategy) == n)
    assert(len(L) == n)
    # we can conventiently compute these by building a 'strategy region graph'
    # which is the graph where two nodes (agents) are connected exactly
    # when both (a) they have the same strategy, and (b) they are adjacent
    # on the lattice (their Manhattan distance is 1).
    #
    # NB using von Neumann neighbourhood (Manhattan distance = 1)
    # here for use with models such as
    # lattice-jointactivity-simcoop-social-noise-cpp-end/model where
    # the von Neumann neighbourhood is used, but note some other models
    # such as lattice-schelling-meanoverlap-network/model used Moore
    # neighbourhood

    # first = time.time()
    # strategy_region_graph = Graph(n)
    # strategy_region_graph.add_edges(
    #         [(i, j) for i in xrange(n) for j in xrange(i)
    #             if Strategy[i] == Strategy[j] and
    #                hackmodel.manhattan_distance(L[i], L[j]) == 1 ] 
    # )
    # components = strategy_region_graph.components()
    # print 'graph strategy regions: ', time.time() - first
    

    # but building the graph is too slow for larger lattices, so
    # use region growing algorithm (no recursion) instead
#    first = time.time()
    (num_regions, max_region_size) = growCulturalRegions(m, n, Strategy, L)
#    print 'nonrecursive region growing strategy regions: ', time.time() - first

#    print 'xxx', 'num_regions =',num_regions, ' components  =',len(components)
#    print 'yyy', 'max_region_size =', max_region_size, ' max compopnent size = ',max(components.sizes())
    # assert(num_regions == len(components))
    # assert(max_region_size == max(components.sizes()))
    
    normalized_num_regions = float(num_regions) / n
    normalized_max_region_size = float(max_region_size) / n
    return (normalized_num_regions, normalized_max_region_size)


# Gets all relevant statistics about the graph, about culture and location of agents
# and writes this information to a file.
def writeStatistics(statsWriter, F, phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b,
                    r, s, t, q, theta, init_random_prob, run,  L, C, Strategy, m, toroidal, network, timestep=-1, lastG = None, lastL = None, lastC = None,
                    componentClusterWriter = None, communityClusterWriter = None, cultureClusterWriter = None, ultrametricWriter =None, correlation = False, differences = None, noise = -1, radius = -1, num_joint_activities = None,
                    pool_multiplier = None, gamestats=None):
   
    n = len(L)
    assert(len(C) == n)

    if init_random_prob == None:
        init_random_prob = "NA"

    pre = [n, m, F, phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, r, s, t, q, theta, init_random_prob]
    
    if radius != -1:
        pre.append(radius)

    if noise != -1:
        pre.append(noise)

    if num_joint_activities != None:
        pre.append(num_joint_activities)

    if pool_multiplier != None:
        pre.append(pool_multiplier)
    
    pre.append(run)

    if timestep != -1:
        pre.append(timestep)     
    
    # unused graph metrics
    avg_path_length = float('nan')
    dia = float('nan')
    avg_degree = float('nan')
    cluster_coeff = float('nan')
    ass = float('nan')
    
    # unused correlation between different spaces
    corr_soc_phy = float('nan')
    corr_soc_cul = float('nan')
    corr_phy_cul = float('nan')
        
    # first = time.time()
    # # Find the cultural clustering and the number of cultures
    # vertex_label, num_cultures, cultures = cultures_in(C, range(n))
    
    # print 'count cultures: ', time.time() - first

    # but, the above is very slow for large n, so instead use numpy:
    first = time.time()
    Carray = np.array(C)
    num_cultures = len(unique_rows(Carray))
    print 'fast count cultures: ', time.time() - first
    # assert(fast_num_cultures == num_cultures)

    num_cultures = float(num_cultures) / n

    # this overall_diversity stat is too slwo for large n, do not do it
    overall_diversity = float('nan')
    # first = time.time()
    # overall_diversity = calcDiversityOfCluster(C, range(n))
    # print 'diversity: ', time.time() - first

    # this size_culture uses vertex_label which is too slow so don't do it
    size_culture = float('nan')
    # first = time.time()
    # # The size of the largest culture can be determined by finding the
    # # vertex label that appears most often
    # size_culture = 0
    # for i in range(len(vertex_label)):
    #     if vertex_label.count(i) > size_culture:
    #         size_culture = vertex_label.count(i)
    
    # size_culture = float(size_culture) / n   
    # print 'largest culture: ', time.time() - first
    
    first = time.time()
    # unused components etc
    num_components = float('nan')
    largest_component = float('nan')
    num_communities = float('nan')
    largest_community = float('nan')
    within_component_diversity = float('nan')
    within_community_diversity = float('nan')
    between_component_diversity = float('nan')
    between_community_diversity = float('nan')
    component_diversity_ratio = float('nan')
    community_diversity_ratio = float('nan')
    social_clustering = float('nan')
    overall_closeness = float('nan')
    social_closeness = float('nan')
    overall_closeness = float('nan')  
    

    physical_closeness = 0.0  
    
    # indices 0 - 9
    stats = [avg_path_length, dia, avg_degree, cluster_coeff, corr_soc_phy, corr_soc_cul, corr_phy_cul, num_cultures, size_culture, overall_diversity, ass]    
    
    # indices 10 - 12
    stats = stats + [num_components, largest_component, within_component_diversity, between_component_diversity, component_diversity_ratio]
    
    # indices 13 - 15
    stats = stats + [num_communities, largest_community, within_community_diversity, between_community_diversity, community_diversity_ratio]
    
    # indices 16 - 19
    stats = stats + [social_clustering, social_closeness, physical_closeness, overall_closeness]
    
    
    
    physicalStability = float('nan')
    socialStability = float('nan')
    culturalStability = float('nan')
        
    stats = stats + [physicalStability, socialStability, culturalStability]


# Don't compute cultural components, too slow (too much memory ) for large n
    num_culture_components = float('nan')

    stats += [num_culture_components]


    first = time.time()
    (num_regions, max_region_size) = calcNumAndMaxSizeOfLatticeCulturalRegions(
                                                                     m, n, C, L)
    print 'cultural regions:', time.time() - first
    stats += [num_regions, max_region_size]



    first = time.time()
    # depends on cooperate=0 and defect=1, and no others,
    # will have to change if others added
    assert(STRATEGY.DEFECT == 1 and STRATEGY.COOPERATE == 0)
    num_defectors = sum(Strategy)
    num_cooperators = n - num_defectors
    print 'num_cooperators = ', num_cooperators
    print 'strategies:', time.time() - first
    stats += [num_cooperators,num_defectors]

    first = time.time()
    (num_strategy_regions,
     max_strategy_region_size) = calcNumAndMaxSizeOfLatticeStrategyRegions(m, n,
                                                   Strategy,L)
    print 'strategy regions:', time.time() - first
    stats += [num_strategy_regions, max_strategy_region_size]
    
            
    # Add public goods game averages
    if gamestats is not None and timestep > 0: # moving avg not valid at t=0
        stats += [gamestats['avg_num_players'],
                  gamestats['avg_num_cooperators'],
                  gamestats['avg_did_cooperate'],
                  gamestats['avg_cooperators_over_players'],
                  gamestats['avg_did_cooperate_over_cooperators'],
                  gamestats['avg_payoff'],
                  gamestats['avg_defector_payoff'],
                  gamestats['avg_cooperator_payoff'],
                  gamestats['avg_mpcr'],
                  gamestats['avg_pool_multiplier']]
    else:
        stats += [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]

    # Add distribution over cultural differences in every step
    if differences is not None:
        stats += differences

    if statsWriter != None:
        statsWriter.writerow(pre + stats)

    return stats







# Write a given degree distribution to a file.
def writeHist(dir, fname, d, n):
    f = open(os.path.join(dir, fname + '.hist'), 'w')
    writer = csv.writer(f)
    writer.writerow(d)
    f.close()
    
    
    
    
    
# Write the network (actually not used this versoin), culture and location of agents to a file.    
def writeNetwork(n, L, C, Strategy, dir, fname):

    first = time.time()
    
    # Write the locations of agents
    f = open(os.path.join(dir, fname + '.L'), 'wb')
    writer = csv.writer(f)
    
    for i in range(len(L)):
        writer.writerow(L[i])
    
    f.close()
    
    # Write the culture of agents
    f = open(os.path.join(dir, fname + '.C'), 'wb')
    writer = csv.writer(f)
    
    for i in range(len(C)):
        writer.writerow(C[i])
    
    f.close()

    # Don't write the .S file any more, cultures_in is far too slow
        
    # # Write the present cultures
    # f = open(os.path.join(dir, fname + '.S'), 'wb')
    # writer = csv.writer(f)

    # i = 0
    # for c in cultures_in(C, range(n))[2]:
    #     writer.writerow([i, c])
    #     i = i+1
    # f.close()

    # Write the present stragies of agents
    f = open(os.path.join(dir, fname + '.Strategy'), 'wb')
    writer = csv.writer(f)
    for i in range(len(Strategy)):
        writer.writerow( [Strategy[i]] )
    f.close()

    print 'writeNetwork: ', time.time() - first

# Determines the svn revisions for all files in this working copy
# and write this information to the provided configuration file.   
def findRevisions(config, path):
    config.add_section('revisions')
    for file in [x for x in index(path) if 'svn' not in x and 'pyc' not in x]:
        #out = commands.getoutput('svn info ' + file)
        #revMatch = re.search('Revision: ([0-9.-]*)', out)
        fileMatch = re.search('[\S]*(\/axelrod\/[\S]*)', file) 
        file = re.sub('^.*/axelrod/', '', file)
#        print 'zzz',file
        out = commands.getoutput('cd ' + path + '; git log --oneline ' + file + '|head -1')
#        print 'xxx',out
        revMatch = re.search('^([0-9a-fA-F]+)', out)
        if revMatch != None:
            config.set('revisions', fileMatch.group(1), revMatch.group(1))
        else:
            config.set('revisions', fileMatch.group(1), 'n/a')
 
 
    
# Lists all files in the directory recursively.  
def index(directory):
    # like os.listdir, but traverses directory trees
    stack = [directory]
    files = []
    while stack:
        directory = stack.pop()
        for file in os.listdir(directory):
            fullname = os.path.join(directory, file)
            files.append(fullname)
            if os.path.isdir(fullname) and not os.path.islink(fullname):
                stack.append(fullname)
    return files


# Determines the diff between the working copy and the latest svn revision
# for all files in this project.
def findDiffs(path, outpath):
    out = commands.getoutput('svn diff ' + path)
    file = open(outpath + 'svn.diff', 'w')
    file.write(out)
    file.close()



# Loads agents' locations, cultures, and the social network (not used this version) from files in
# the directory given by the argument path.
def loadNetwork(path, network = True, end = False):   
    L = list()
    reader = csv.reader(open(path + '.L'))                
    for row in reader:
        L.append((int(row[0]), int(row[1])))
    
    C = list()
    reader = csv.reader(open(path + '.C'))          
    for row in reader:
        C.append(array([int(x) for x in row]))
        
    if end:    
        infile = open(path + '.Tend', "r")
        tend = infile.readline()
    else:
        tend = 0

    Strategy = list()
    reader = csv.reader(open(path + '.Strategy'))
    for row in reader:
        Strategy.append(int(row[0]))
        
        
    if os.path.exists(path + '.D') and os.path.isfile(path + '.D'):
        reader = csv.reader(open(path + '.D'))
        for row in reader:
            D = [int(x) for x in row]

    else:
        D = None
        
    
    return L, C, Strategy, D, tend

# Load average public goods game stats from file in directory given by filename
# with format
# avg_num_players = 1.12345
# etc.
# returns dict name:value e.g. gamestats["avg_num_players"] = 1.123
def loadGamestats(filename):
    gamestats = {}
    for line in open(filename):
        name, var = line.partition("=")[::2]
        gamestats[name.strip()] = float(var)
    return gamestats

    
# Runs the C++ version of the model.
def modelCpp(L,C,Strategy,tmax,n,m,F,q,r,s,toroidal,network,t,directedMigration,
             phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, modelpath,
             r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, modelCallback = None, noise = -1, end = False, k = 0.01, theta = 0.0,
             no_migration = False, radius = -1, num_joint_activities=None,
             pool_multiplier= None, strategy_update_rule = None,
             culture_update_rule = None):
    
    if toroidal: 
        toroidalS = '1'
    else:
        toroidalS = '0'
        
    if network: 
        networkS = '1'
    else:
        networkS = '0'         
        
    if directedMigration: 
        directedMigrationS = '1'
    else:
        directedMigrationS = '0'

    tmpdir = os.tempnam(None, 'exp')
    os.mkdir(tmpdir)
    writeNetwork(n, L, C, Strategy, tmpdir, 'tmp')
    
    # If all cells are occupied, there cannot need to be any migration
    if n == m*m:
        s = 0.0

    if no_migration:  # if no_migration is set, disable migration 
        s = 0.0

#    print 'xxx modelCpp tmax = ', tmax
    
    options = [tmax,n,m,F,q,r,s,toroidalS,networkS,t,directedMigrationS,
               phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b, 
               r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, k,
               tmpdir + '/' + 'tmp', theta]
    

    if radius != -1:
        options.append(radius)

    if noise != -1:
        options.append(noise)

    if num_joint_activities != None:
        options.append(num_joint_activities)
    if pool_multiplier != None:
        options.append(pool_multiplier)
    if strategy_update_rule != None:
        assert(strategy_update_rule == UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER or strategy_update_rule == UPDATE_RULE.UPDATE_RULE_FERMI or strategy_update_rule == UPDATE_RULE.UPDATE_RULE_MODALBEST)
        options.append(strategy_update_rule)
    if culture_update_rule != None:
        assert(culture_update_rule == UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER or culture_update_rule == UPDATE_RULE.UPDATE_RULE_FERMI or culture_update_rule == UPDATE_RULE.UPDATE_RULE_MODAL or culture_update_rule == UPDATE_RULE.UPDATE_RULE_MODALBEST)
        options.append(culture_update_rule)
        
    
    if modelCallback is not None:
        f = open(tmpdir + '/tmp.T', 'wb')
        writer = csv.writer(f)
        writer.writerow([len(modelCallback.call_list)])
        writer.writerow(modelCallback.call_list)        
        f.close()        
        #options += modelCallback.call_list            
    
    first = time.time()
    output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]))
    #XXX output = commands.getoutput(modelpath + ' ' + ' '.join([str(x) for x in options]) + ' | tee -a model_stdout.txt')
    print output
    print 'model: ', time.time() - first
    
    if modelCallback is not None:
        for iteration in modelCallback.call_list:
            if (iteration != tmax and
                os.path.exists(tmpdir+'/tmp-'+str(iteration)+'.L')):
                L2, C2, Strategy2, D2, tmp = loadNetwork(tmpdir + '/tmp-' + str(iteration), network, end = False)
                gamestats2 = loadGamestats(tmpdir + '/tmp-' + str(iteration) + '.Gamestats')
                if D2 is not None:
                    modelCallback.call(L2, C2, Strategy2, iteration, D2, gamestats2)
                else:
                    modelCallback.call(L2, C2, Strategy2, iteration, gamestats2)
    
    L2, C2, Strategy2, D2, tend = loadNetwork(tmpdir + '/tmp', network, end = end)
    gamestats2 = loadGamestats(tmpdir + '/tmp.Gamestats')
    
    for filename in glob.glob(os.path.join(tmpdir, "*")):
        os.remove(filename)
    os.rmdir(tmpdir)
    
    if D2 is not None:
        return L2, C2, Strategy2, D2, gamestats2
    elif end:
        return L2, C2, Strategy2, tend, gamestats2
    else:
        return L2, C2, Strategy2, tmax, gamestats2    
    
    
