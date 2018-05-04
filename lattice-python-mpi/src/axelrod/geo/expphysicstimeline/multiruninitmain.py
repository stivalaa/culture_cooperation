#  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
#  Copyright (C) 2014 Alex Stivala <stivalaa@unimelb.edu.au>
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

# This version is also simplified to remove stuff to do with social networks
# etc. for use with simpler Axelrod type models on lattice to not waste
# time/memory on these computations. Note the columns are left in output
# however to maintain compatibility with shell scripts etc. from more
# complex versions.

# This version (expphysicstimeline/multiruninitmain.py) is just like
# expphysicstimeline/main.py only instead of doing random initial state
# at each of the (50) runs, it does the random initial state once
# only for each set of parameters, then does all (50) runs from stame
# initial state (in same MPI task). This makes getting error bars etc.
# for given number of initial culture components easier as now only variation
# in stats on initial state in equilbirium not also initial state.
# For better use of parallel tasks, this version distributes the 50 runs
# among potentially different tasks, rather than running 50 repeats of
# the same parameter set sequentially in one task.

import os, re, glob, sys


import warnings # so we can suppress the annoying tempnam 'security' warning

import sys
import os,errno
sys.path.append( os.path.abspath(os.path.dirname(sys.argv[0])).replace('/expphysicstimeline','') )

from numpy import array
from random import randrange,sample,random
import math
from time import strftime,localtime,sleep

import csv
#import model

import lib
import ConfigParser
import commands

from mpi4py import MPI

import neutralevolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpi_processes = comm.Get_size()


def writeConfig(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list, 
             n_list, q_list, 
             directedMigration, cppmodel = None, time_list = None, 
             theta_list = None, evolved_init_culture = False,
             evolved_init_culture2 = False,
             prototype_init_culture = False,
             trivial_ultrametric_init_culture = False,
             init_culture_csv_file = None,
             init_random_prob_list = None,
             resumeFrom = None, 
             no_migration = False,
             cluster_k =None,
             ef2_r = None,
             radius_list = None,
             noise_list = None,
             num_joint_activites_list = None,
             pool_multiplier_list = None,
             strategy_update_rule = None,
             culture_update_rule = None,
             snapshot_time_list = None,
             zero_init_culture = False,
             init_strategy = lib.INIT_STRATEGY.RANDOM
             ):
    # Write out current configuration       
    for n in n_list:
        try:
            os.mkdir(dir + str(n))
        except OSError,e:
            if e.errno == errno.EEXIST:
                pass
            
        # Save the diff between the current working copy and the latest svn revision
        lib.findDiffs(scriptpath, dir + str(n) + '/')

        # Write out the parameters for this experiment
        config = ConfigParser.RawConfigParser()

        config.add_section('environment')
        config.set('environment', 'starttime', strftime("%d%b%Y-%H:%M:%S", localtime()))
        config.set('environment', 'mpiname' ,comm.Get_name())
        config.set('environment', 'mpiprocessorname' ,MPI.Get_processor_name())
        config.set('environment', 'mpiprocesses' ,mpi_processes)
        config.set('environment', 'mpirank', rank)

        # Determine the svn revisions of all files and write them into the config file
        lib.findRevisions(config, scriptpath)

        # Write out all parameters
        config.add_section('paras')
        if cppmodel is not None:
            config.set('paras', 'cpp', commands.getoutput(cppmodel + ' -v'))
        else:
            config.set('paras', 'cpp', False)
        config.set('paras', 'runs', str(runs))
        config.set('paras', 'tmax', str(tmax))
        config.set('paras', 'F', str(F))
        config.set('paras', 'm', str(m))
        config.set('paras', 'toroidal', str(toroidal))
        config.set('paras', 'network', str(network))
        config.set('paras', 'beta_p_list', ','.join([str(x) for x in beta_p_list]))
        config.set('paras', 'beta_s_list', ','.join([str(x) for x in beta_s_list]))        
        config.set('paras', 'n', str(n))
        config.set('paras', 'directed_migration', str(directedMigration))
        config.set('paras', 'q_list', ','.join([str(x) for x in q_list]))
        config.set('paras', 'theta_list', ','.join([str(x) for x in theta_list]))
        if time_list is not None:
            config.set('paras', 'time_list', ','.join([str(x) for x in time_list]))
        config.set('paras', 'evolved_init_culture', str(evolved_init_culture))
        config.set('paras', 'evolved_init_culture2', str(evolved_init_culture2))
        config.set('paras', 'prototype_init_culture', str(prototype_init_culture))
        config.set('paras', 'trivial_ultrametric_init_culture', str(trivial_ultrametric_init_culture))
        config.set('paras', 'zero_init_culture', str(zero_init_culture))
        config.set('paras', 'init_strategy', "RANDOM" if init_strategy == lib.INIT_STRATEGY.RANDOM else ("STRIPE" if init_strategy == lib.INIT_STRATEGY.STRIPE else ("CHESSBOARD" if init_strategy == lib.INIT_STRATEGY.CHESSBOARD else "*ERROR*")))
        if prototype_init_culture:
            config.set('paras', 'cluster_k', str(cluster_k))
        if evolved_init_culture2:
            config.set('paras', 'ef2_r', str(ef2_r))
        if evolved_init_culture or evolved_init_culture2 or prototype_init_culture:
            config.set('paras', 'init_random_prob_list', ','.join([str(x) for x in init_random_prob_list]))
        if init_culture_csv_file is not None:
            config.set('paras', 'read_init_culture', init_culture_csv_file)
        if resumeFrom != None:
            config.set('paras', 'resumeFrom', str(resumeFrom))
            writemode = 'ab'  # append to config file
        else:
            writemode = 'wb'  # overwrite config file
        config.set('paras', 'no_migration', str(no_migration));
        config.set('paras', 'radius_list', ','.join([str(x) for x in radius_list]))
        config.set('paras', 'noise_list', ','.join([str(x) for x in noise_list]))
        config.set('paras', 'num_joint_activities_list', ','.join([str(x) for x in num_joint_activities_list]))
        config.set('paras', 'pool_multiplier_list', ','.join([str(x) for x in pool_multiplier_list]))
        config.set('paras', 'strategy_update_rule', "BESTTAKESOVER" if strategy_update_rule == lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER else ("FERMI" if strategy_update_rule == lib.UPDATE_RULE.UPDATE_RULE_FERMI else ("MODALBEST" if strategy_update_rule == lib.UPDATE_RULE.UPDATE_RULE_MODALBEST else "*ERROR*")))
        config.set('paras', 'culture_update_rule', "BESTTAKESOVER" if culture_update_rule == lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER else ("FERMI" if culture_update_rule == lib.UPDATE_RULE.UPDATE_RULE_FERMI else ("MODAL" if culture_update_rule == lib.UPDATE_RULE.UPDATE_RULE_MODAL else ("MODALBEST" if culture_update_rule == lib.UPDATE_RULE.UPDATE_RULE_MODALBEST else "*ERROR*"))))
        if snapshot_time_list is not None:
            config.set('paras', 'snapshot_time_list', ','.join([str(x) for x in snapshot_time_list]))
        config.write(open(dir + str(n) + '/' + 'parameter' + str(rank) + '.cfg', writemode))   





class ModelCallback:
    def __init__(self, graph_list, statsWriter, F, phy_mob_a, phy_mob_b, 
                 soc_mob_a, soc_mob_b, r, s, t, q, theta, init_random_prob, i,
                 toroidal,network,
                 componentClusterWriter, communityClusterWriter, cultureClusterWriter, ultrametricWriter, radius, noise, num_joint_activities, pool_multiplier, strategy_udpate_rule, culture_update_rule, snapshot_time_list, dir, n):
        self.call_list = graph_list
        self.statsWriter = statsWriter
        self.F = F
        self.phy_mob_a = phy_mob_a
        self.phy_mob_b = phy_mob_b
        self.soc_mob_a = soc_mob_a
        self.soc_mob_b = soc_mob_b
        self.r = r
        self.s = s
        self.t = t
        self.q = q
        self.theta = theta
        self.init_random_prob = init_random_prob
        self.i = i
        self.toroidal = toroidal
        self.network = network
        self.lastG = None
        self.lastL = None
        self.lastC = None
        self.componentClusterWriter = componentClusterWriter
        self.communityClusterWriter = communityClusterWriter
        self.cultureClusterWriter = cultureClusterWriter
        self.ultrametricWriter = ultrametricWriter
        self.radius = radius
        self.noise = noise
        self.num_joint_activities = num_joint_activities
        self.pool_multiplier = pool_multiplier
        self.strategy_update_rule = strategy_update_rule
        self.culture_update_rule = culture_update_rule
        self.snapshot_time_list = snapshot_time_list
        self.dir = dir
        self.n = n
        
    def call(self, L, C, Strategy, iteration, gamestats):                        
        # Get statistics for this run
        lib.writeStatistics(self.statsWriter, self.F, self.phy_mob_a, 
                              self.phy_mob_b, self.soc_mob_a, self.soc_mob_b,
                              self.r,self.s,self.t,self.q,self.theta,
                              self.init_random_prob,self.i,L,C,Strategy,m,
                              self.toroidal,
                              self.network,iteration,self.lastG, self.lastL, self.lastC, 
                              componentClusterWriter = self.componentClusterWriter, 
                              communityClusterWriter = self.communityClusterWriter, 
                              cultureClusterWriter= self.cultureClusterWriter,
                              ultrametricWriter = self.ultrametricWriter,
                              correlation = True,
                              radius = self.radius,
                              noise = self.noise,
                              num_joint_activities = self.num_joint_activities,
                              pool_multiplier = self.pool_multiplier,
                              gamestats=gamestats
                            )
        
        # for run number 0, write all data if this iteration is in snapshot list
        if self.i == 0 and iteration in self.snapshot_time_list:
            filename = 'results' + '-n' + str(self.n) + '-q' + str(self.q) + '-beta_p' + str(self.phy_mob_b) + '-beta_s' + str(self.soc_mob_b) + '-theta' + str(self.theta) + '-init_random_prob' + str(self.init_random_prob) + '-radius' + str(self.radius) + '-num_joint_activities' + str(self.num_joint_activities) + '-pool_multiplier' + ('%.8f' % self.pool_multiplier) + '-noise' + ('%.8f' % self.noise)
            lib.writeNetwork(self.n,L,C,Strategy, self.dir+str(self.n), filename + '-' + str(iteration))

        
        self.lastL = list(L)
        
        self.lastC = list()
        for c in C:
            self.lastC.append(c.copy())



def scenario(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list, 
             n_list, q_list, time_list,
             directedMigration=True, cppmodel = None, end = False,
             evolved_init_culture = False, 
             evolved_init_culture2 = False,
             prototype_init_culture = False,
             trivial_ultrametric_init_culture = False,
             theta_list = None,
             initial_culture = None,
             init_culture_csv_file = None,
             init_random_prob_list = None,
             cluster_k = None,
             ef2_r = None,
             radius_list = None,
             noise_list = None,
             num_joint_activities_list = None,
             pool_multiplier_list = None,
             strategy_update_rule = None,
             culture_update_rule = None,
             resumeFrom = None,
             snapshot_time_list = None,
             zero_init_culture = False,
             init_strategy = lib.INIT_STRATEGY.RANDOM):   

    assert not (initial_culture != None and evolved_init_culture)
    assert not (initial_culture != None and evolved_init_culture2)
    assert not (initial_culture != None and prototype_init_culture)
    assert not (initial_culture != None and trivial_ultrametric_init_culture)
    assert not (initial_culture != None and zero_init_culture)
    assert not (evolved_init_culture and evolved_init_culture2)
    assert not (evolved_init_culture and prototype_init_culture)
    assert not (evolved_init_culture2 and prototype_init_culture)
    assert not (evolved_init_culture and trivial_ultrametric_init_culture)
    assert not (evolved_init_culture2 and trivial_ultrametric_init_culture)
    assert not (prototype_init_culture and trivial_ultrametric_init_culture)
    assert not (not prototype_init_culture and cluster_k != None)
    assert not (prototype_init_culture and cluster_k == None)
    assert not (not evolved_init_culture2 and ef2_r != None)
    assert not (evolved_init_culture2 and ef2_r == None)
    assert [evolved_init_culture, evolved_init_culture2,
            trivial_ultrametric_init_culture, zero_init_culture].count(True) <= 1
    # TODO probably should have had an integer init_culture_type or something
    # instead of now having all these booleans...
    if not (evolved_init_culture or evolved_init_culture2 or prototype_init_culture or trivial_ultrametric_init_culture):
        init_random_prob_list = [None] # not used if not evolved_init_culture

    assert(init_strategy in [lib.INIT_STRATEGY.RANDOM, lib.INIT_STRATEGY.STRIPE, lib.INIT_STRATEGY.CHESSBOARD])
    
    if cppmodel is not None:
        print "Using c++ version with ", cppmodel
    else:
        print "Using python version"
        
    if end:
        time_list = []
        snapshot_time_list = []
    
    try:
        os.mkdir(dir)       
    except OSError,e:
        if e.errno == errno.EEXIST:
            pass
      

#    print 'xxx scenario tmax = ', tmax
    
    writeConfig(scriptpath, runs, dir, tmax, F, m, toroidal, network, 
             beta_p_list, beta_s_list,
             n_list, q_list, 
             directedMigration, cppmodel, time_list, theta_list,
             evolved_init_culture, evolved_init_culture2,
             prototype_init_culture,
             trivial_ultrametric_init_culture,
             init_culture_csv_file,
             init_random_prob_list, resumeFrom, no_migration, cluster_k, ef2_r,
             radius_list, noise_list, num_joint_activities_list,
             pool_multiplier_list, strategy_update_rule, culture_update_rule,
             snapshot_time_list, zero_init_culture, init_strategy)
    
        

    statsWriter = {}  # dict indexec by n
    componentClusterWriter = {} # dict indexed by n
    communityClusterWriter = {} # dict indexed by n
    cultureClusterWriter = {} # dict indexed by n
    ultrametricWriter = {} # dict indexed by n

    param_list = [] # list of parameter tuples: build list then run each
    param_count = 0
    for n in n_list:
        # Create the file into which statistics are written
        results_prefix = '/results' + str(rank)
        resultsfilename = dir + str(n) + results_prefix + '.csv'
        if resumeFrom != None:
            print 'Resuming from ', resumeFrom
            print 'Appending results to ', resultsfilename
            csvwritemode = 'a'  # append if resuming 
        else:
            print 'Clean start'
            print 'Writing results to ', resultsfilename
            csvwritemode = 'w'  # overwite if staring from beginning
        file = open(resultsfilename, csvwritemode)
        statsWriter[n] = csv.writer(file)
        
        componentClusterWriterFile= open(dir + str(n) + results_prefix + '-size_x_div-components.csv', csvwritemode)
        communityClusterWriterFile= open(dir + str(n) + results_prefix + '-size_x_div-communities.csv', csvwritemode)
        cultureClusterWriterFile = open(dir + str(n) + results_prefix +'-cultures.csv', csvwritemode)
        ultrametricWriterFile = open(dir + str(n) + results_prefix + '-ultrametricity.csv', csvwritemode)

        componentClusterWriter[n] = csv.writer(componentClusterWriterFile)
        communityClusterWriter[n] = csv.writer(communityClusterWriterFile)
        cultureClusterWriter[n] = csv.writer(cultureClusterWriterFile)
        ultrametricWriter[n] = csv.writer(ultrametricWriterFile)

        for run_number in xrange(runs):
            for q in q_list:
                for beta_p in beta_p_list:
                    for beta_s in beta_s_list:              
                        for theta in theta_list:
                            for init_random_prob in init_random_prob_list:
                                for radius in radius_list:
                                    for num_joint_activities in num_joint_activities_list:
                                        for pool_multiplier in pool_multiplier_list:
                                            for noise in noise_list:
                                                param_count += 1
                                                if resumeFrom is not None:
                                                    # 2.0e-07 below seems necesary to get resumeFrom to work on noise, 1e-08 fails
                                                    if n == resumeFrom[0] and q == resumeFrom[1] and  beta_p == resumeFrom[2] and beta_s == resumeFrom[3] and theta == resumeFrom[4] and init_random_prob == resumeFrom[5] and radius == resumeFrom[6] and num_joint_activities == resumeFrom[7] and (abs(pool_multiplier - resumeFrom[8]) < 2.0e-07) and  (abs(noise - resumeFrom[9]) < 2.0e-07) and run_number == resumeFrom[10]:
                                                        resumeFrom = None
                                                    else:
                                                        continue                               

                                                param_list.append((n, q, beta_p, beta_s, theta, init_random_prob, radius, num_joint_activities, pool_multiplier, noise, run_number))



    print len(param_list),'of total',param_count,'models to run'
    print int(math.ceil(float(len(param_list))/mpi_processes)),' models per MPI task'
    if not end:
        print 'time series: writing total',len(time_list)*param_count,'time step records'

    # now that we have list of parameter tuples, execute in parallel
    # using MPI: each MPI process will process 
    # ceil(num_jobs / mpi_processes) of the parmeter tuples in the list
    num_jobs = len(param_list)
    job_i = rank
    while job_i < num_jobs:
        (n, q, beta_p, beta_s, theta, init_random_prob, radius, num_joint_activities, pool_multiplier, noise, run_number) = param_list[job_i]

        # In this format, the tuple after 'rank 0: '
        # can be cut&pasted into resumeFrom: command line
        sys.stdout.write('rank %d: %d,%d,%f,%f,%f,%s,%d,%d,%f,%f,%d\n' %
                         (rank,n,q,beta_p,beta_s,theta,str(init_random_prob),radius,num_joint_activities,pool_multiplier,noise,run_number))

        assert(n <= m**2) # lattice must have enough positions for agents

        if end:
            if (noise <= 1e-6):
              tmax = max(tmax, n*1000000)
            else:
              tmax = max(tmax, n*100000)

#XXX        tmax = min(tmax, 100000000) #  limit to 10^8 iterations at most


#        print 'xxx scenario(2) tmax = ', tmax
        
        # Using the same intalization for each run with same parameters, assumign
        # that run_number is the outermost loop in building the parameter tuple
        # list and rusn from 0..runs
        # taking advnatage of the fact that the intializztion data is
        # written to filesystem which the C++ code reads anyway, on shared
        # filesystem.

        initfilename = 'results' + '-n' + str(n) + '-q' + str(q) + '-beta_p' + str(beta_p) + '-beta_s' + str(beta_s) + '-theta' + str(theta) + '-init_random_prob' + str(init_random_prob) + '-radius' + str(radius) + '-num_joint_activities' + str(num_joint_activities) + '-pool_multiplier' + ('%.8f' % pool_multiplier) + '-noise' + ('%.8f' % noise)  + '-' + str(0)
        initdirname = dir+str(n)

        if run_number == 0:
            # Set random positions of agents
            startL = [(x,y) for x in range(m) for y in range(m)]
            L = list()
            for j in range(n):
                idx = randrange(len(startL))
                L.append(startL[idx])
                del startL[idx]

            # Create culture vector for all agents
            if initial_culture:
                C = initial_culture
                assert(n == len(C))
                assert(F == len(C[0]))

            elif evolved_init_culture:
              # 'evolve' culture vectors, then perturb each element
              # with probability init_random_prob, picking a random value for
              # that element,
              initialC = [randrange(q) for k in range(F)]
              C = neutralevolution.ef(initialC, q, [],
                                      int(math.ceil(math.log(n,2))),
                                      1.0, 1.0)[:n]
              C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

            elif evolved_init_culture2:
              # 'evolve' culture vectors (with ef2 function not original ef),
              # mutating ef2_r  traits (equiv to ef2() if r=1,
              # efault original ef2() if r=F/2) each step (not one trait)
              # then perturb each element
              # with probability init_random_prob, picking a random value for
              # that element,
              initialC = [randrange(q) for k in range(F)]
              C = neutralevolution.ef2(initialC, q, [],
                                      int(math.ceil(math.log(n,2))),
                                      ef2_r)[:n]
              C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

            elif prototype_init_culture:
              # 'evolve' culture vectors by 'reverse k-means' generating k
              # prototype vectors then randomly createding other near htme by
              # mutating 1/2 of all traits each step ,
              # then perturb each element
              # with probability init_random_prob, picking a random value for
              # that element,
              initialC = [randrange(q) for k in range(F)]
              C = neutralevolution.prototype_evolve(F, q, n, cluster_k, F/2)
              C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

            elif trivial_ultrametric_init_culture:
              # create ultrametric init culture that is perfectly ultrametric
              # but trivial in that all vectors have the same hamming distance
              C = neutralevolution.trivial_ultrametric(F, q, n)
              C = [array(v) for v in neutralevolution.perturb(array(C), q, init_random_prob).tolist()]  # have to convert array to list and then each inner list back to array to from 2d array for pertrub() to list of 1d arrays

            elif zero_init_culture:
              # all culture vectors are same constant (zero), used for
              # testing strategy cooperation when effect of culture is
              # not present since all agents have identicial culture
              # vectors
              C = [array([0 for k in range(F)]) for l in range(n)]

            else: # uniform random culture vectors
              C = [array([randrange(q) for k in range(F)]) for l in range(n)]

            if init_strategy == lib.INIT_STRATEGY.RANDOM:
                # uniform rand strategies
                Strategy = [randrange(2) for l in range(n)]
            elif init_strategy == lib.INIT_STRATEGY.STRIPE:
                # one side of lattice is cooperate, other half defect
                assert(m % 2 == 0)
                Strategy = [lib.STRATEGY.DEFECT for i in range(n)]
                for i in range(n):
                    if L[i][0] < m/2:
                        Strategy[i] = lib.STRATEGY.COOPERATE
            elif init_strategy == lib.INIT_STRATEGY.CHESSBOARD:
                # checkerboard pattern
                sys.stderr.write('TODO not implemented chessobard init strategy')
                sys.exit(1)
            else:
                sys.stderr.write('bad init_strategy')
                sys.exit(1)

            # write initial conditions to a file
            lib.writeNetwork(n,L,C,Strategy, initdirname, initfilename)


        graphCallback = ModelCallback(time_list, statsWriter[n], F, 
            1, beta_p, 1, beta_s, 
            1, 1, -1, q, theta, init_random_prob, run_number, toroidal, network,
            componentClusterWriter[n], communityClusterWriter[n], cultureClusterWriter[n], ultrametricWriter[n], radius, noise, num_joint_activities, pool_multiplier, strategy_update_rule, culture_update_rule, snapshot_time_list, dir, n)


        # read back the culture, lattice etc. from file written on single
        # (run 0) initialization if this is not run 0
        # Although relying on the fact 
        # that run_number is the outermost loop in building the parameter tuple
        # list and rusn from 0..runs so run 0 will always be first, if there
        # are enough tasks (or few enough parameter tuples), it is possible
        # that run 0 and a subsequent run(s) will actually be scheduled
        # simultaneously, so we might have to have the subsequent runs
        # wait a little while if run 0 has not started and written the
        # initialization data yet.
        if run_number > 0:
            done = False
            retry = 0
            while not done:
                try: 
                    L, C, Strategy, D, tend = lib.loadNetwork(os.path.join(initdirname,
                                                                  initfilename))
                    done = True
                except IOError as e:
                    retry += 1
                    sys.stdout.write('rank %d run %d retry %d waiting for initial data' 
                                     % (rank,run_number,retry))
                    sleep(5)


        # write initial stats to stats files
        # (original version did not do this, now have to account for 
        # iteratino 0 in results for for end (run to equilibrium) runs,
        # rather than just final iteration number when equilibrium reached)
        if end: # in time (not equilibrium) mode, already writes at iter=0
            graphCallback.call(L, C, Strategy, 0, None)

        # Run the model
        L2, C2, Strategy2, tend, gamestats2 = lib.modelCpp(L,C,Strategy,tmax,n,m,F,q,1,1,toroidal,network,-1,
                              directedMigration, 1, beta_p, 1, beta_s, cppmodel,
                              1, 1, 1, beta_p, 1, beta_s, graphCallback, 
                              end = end, 
                              k = 0.01, theta = theta, no_migration =no_migration, radius = radius, num_joint_activities = num_joint_activities, pool_multiplier = pool_multiplier, noise = noise, strategy_update_rule = strategy_update_rule, culture_update_rule = culture_update_rule) 

        if end:
            graphCallback.call(L2, C2, Strategy2,  tend, gamestats2)
            iterno = tend
        else:
            graphCallback.call(L2, C2, Strategy2 , tmax,gamestats2)
            iterno = tmax


        file.flush()
        componentClusterWriterFile.flush()
        communityClusterWriterFile.flush()
        cultureClusterWriterFile.flush()
        ultrametricWriterFile.flush()

        if run_number == 0:
          # write final conditions to a file
          filename = 'results' + '-n' + str(n) + '-q' + str(q) + '-beta_p' + str(beta_p) + '-beta_s' + str(beta_s) + '-theta' + str(theta) + '-init_random_prob' + str(init_random_prob) + '-radius' + str(radius) + '-num_joint_activities' + str(num_joint_activities) + '-pool_multiplier' + ('%.8f' % pool_multiplier) + '-noise' + ('%.8f' % noise)
          lib.writeNetwork(n,L2,C2,Strategy2, dir+str(n), filename + '-' + str(iterno))

        job_i += mpi_processes

    file.close()
    componentClusterWriterFile.close()
    communityClusterWriterFile.close()
    cultureClusterWriterFile.close()
    ultrametricWriterFile.close()


if __name__ == '__main__':   

    # tmpdir() annoyingly gives 'security' warning on stderr, as does
    # tmpnam(), unless we add these filterwarnings() calls.
    warnings.filterwarnings('ignore', 'tempdir', RuntimeWarning)
    warnings.filterwarnings('ignore', 'tempnam', RuntimeWarning)

    try:
        import psyco
        #psyco.log()
        psyco.full()
    except ImportError:
        print "Psyco not installed or failed execution."    

    # Find the path to the source code
    scriptpath = os.path.abspath(os.path.dirname(sys.argv[0])).replace('/geo/expphysicstimeline', '/')
  
    #q_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,30,35,40,50,70,100] 
    #q_list = [2,3,5,8,10,15,20,25,30,35,40,50,70,100] 
    #q_list = [10]
    #q_list = [2,5,10,15,30,100]
    #q_list = [2,5,10 ]
    #q_list = [  15    ]
    #q_list = [2,5,10,   30,100]
    q_list = [30, 75]
  
    beta_p_list = [10]
    #beta_s_list = [1,10]
    # only use beta_s =1 , larger is too slow to converge
    beta_s_list = [1]
   
    #theta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #theta_list = [0.0  ,  0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,  0.35,  0.4 , 0.45,  0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85, 0.9 ,  0.95, 1.0] 
    #theta_list = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0]
    theta_list = [0]
    
    # probability of each element of initial culture vector being random
    # rather than inherited from parent if evolved_init_culture is used
    # so 1 is same as original completely random init
    #init_random_prob_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    init_random_prob_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # von Neumann neighbourhood radius
    #radius_list = [ 1,2,3,4,5,6,7,8,9,10 ]
    #radius_list =  [ 1,        6,      10 ]
    radius_list =  [            2          ]

    # number of joint activiities (public goods games)
    num_joint_activities_list = [5]

    # public goods game pool multiplier
    #pool_multiplier_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2.0]
    #pool_multiplier_list = [0.2, 0.4, 0.6, 0.8, 1.0]
    pool_multiplier_list = [          0.6          ]
    
    # rate of noise
    #noise_list = [ 0, 1.0E-6, 1.584893E-6, 2.511886E-6, 3.981072E-6, 6.309573E-6, 1.0E-5, 1.584893E-5, 2.511886E-5, 3.981072E-5, 6.309573E-5, 1.0E-4, 1.584893E-4, 2.511886E-4, 3.981072E-4, 6.309573E-4, 0.001, 0.001584893, 0.002511886, 0.003981072, 0.006309573, 0.01, 0.01584893, 0.02511886, 0.03981072, 0.06309573, 0.1 ]
    noise_list = [ 0, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01 ]


    runs = 50
    
    #tmax = 1000000  # 1e06
    #tmax = 100000000 # 1e08
    tmax = 1000000000 # 1e09

    #time_list = [0, 5000, 10000, 20000, 50000, 100000, 500000, 1000000, 2000000, 5000000, 10000000, 50000000, 100000000]
    #time_list = [0, 5000, 10000, 20000, 50000,  100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000]
    #time_list = [0,100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
#    time_list = [0,50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,950000,1000000]
    #time_list = [0, 5000000, 10000000, 15000000, 20000000, 25000000, 30000000, 35000000, 40000000, 45000000, 50000000, 55000000, 60000000, 65000000, 70000000, 75000000, 80000000, 85000000, 90000000, 95000000, 100000000]
    #time_list = range(0,1001000,10000)      # 1e06, 100 time points
    #time_list = range(0,100100000,1000000)   # 1e08, 100 time points
    time_list = range(0,1001000000,10000000)   # 1e09, 100 time points

    # write all data (culture, strategy, lattice) every 10th time_list step
    # (and not at first and last, already done specifically)
    snapshot_time_list = [time_list[i] for i in range(1, len(time_list)-1, len(time_list)/10)]
    
    m = 25
    F = 5

    
    directedMigration = False
    
    cppmodel = None
    
    n_list = []
    
    end = False
    
    evolved_init_culture = False
    evolved_init_culture2 = False
    prototype_init_culture = False
    trivial_ultrametric_init_culture = False
    zero_init_culture = False
    init_culture_csv_file = None
    initial_culture= None
    resumeFrom = None
    no_migration = True # default is to have migration
    cluster_k = None
    ef2_r = None

    strategy_update_rule = lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER
    culture_update_rule = lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER    

    init_strategy = lib.INIT_STRATEGY.RANDOM
    
    for arg in sys.argv[1:]:
        if arg == 'end':
            end = True
            print "Run until convergence"
        
        elif arg == 'undirected':
            directedMigration = False   
            print 'Undirected/Random migration'

        elif arg == 'evolved_init_culture': # instead of uniform init C vectors
            evolved_init_culture = True
            print 'evolutionary initial culture vectors'

        elif (arg == 'evolved_init_culture2'  # instead of uniform init C vectors
              or (len(arg.split(':')) == 2 and
                  arg.split(':')[0] == 'evolved_init_culture2')):
            evolved_init_culture2 = True
            print 'evolutionary initial culture vectors (new version ef2)'
            if (len(arg.split(':')) == 2):
                # value of r the number of traits to change each step
                ef2_r = int(arg.split(':')[1])

        elif (arg == 'prototype_init_culture' #instead of uniform init C vectors
              or (len(arg.split(':')) == 2 and
                  arg.split(':')[0] == 'prototype_init_culture')):
            prototype_init_culture = True
            print 'reverse k-means prototype based initial culture vectors'
            if (len(arg.split(':')) == 2):
                # value of k the number of prototypes
                cluster_k = int(arg.split(':')[1])
            else:
                cluster_k = 3

        elif arg == 'trivial_ultrametric_init_culture': # instead of uniform init C vectors, trivial but perfectly ultrametric
            trivial_ultrametric_init_culture = True
            print 'trivial ultrametric initial culture vectors'

        elif arg == 'zero_init_culture': # all vectors zero instead of random
            zero_init_culture = True
            
        elif arg == 'no_migration':
            no_migration = True
            print 'No migration'

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'F'):
              # set value of F, dimension of culture vectors (default 5)
              F = int(arg.split(':')[1])

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'm'):
              # set value of m, dimension of lattice (default 25)
              m = int(arg.split(':')[1])

        elif (len(arg.split(':')) == 2 and  
              arg.split(':')[0] == 'read_init_culture'):
             # read_init_culture:culture.csv
             # read initial culture from CSV file
             # in this mode we get the values of q, F, and n from the file
             # (resp. max number of different values in a column,
             # number of columns, number of rows)
             init_culture_csv_file = arg.split(':')[1]

        elif (len(arg.split(':')) == 2 and
              arg.split(':')[0] == 'strategy_update_rule'):
            # strategy update rule
            if arg.split(':')[1] == "best_takes_over":
                strategy_update_rule = lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER
            elif arg.split(':')[1] == "fermi":
                strategy_update_rule = lib.UPDATE_RULE.UPDATE_RULE_FERMI
            elif arg.split(':')[1] == "modal":
                 sys.stderr.write('modal update rule not available for strategy\n')
            elif arg.split(':')[1] == "modalbest":
                 strategy_update_rule = lib.UPDATE_RULE.UPDATE_RULE_MODALBEST
            else:
                sys.stderr.write('bad update rule ' + arg.split(':')[1]+'\n')
                sys.exit(1)

        elif (len(arg.split(':')) == 2 and
              arg.split(':')[0] == 'culture_update_rule'):
            # culture vector update rule
            if arg.split(':')[1] == "best_takes_over":
                culture_update_rule = lib.UPDATE_RULE.UPDATE_RULE_BESTTAKES_OVER
            elif arg.split(':')[1] == "fermi":
                culture_update_rule = lib.UPDATE_RULE.UPDATE_RULE_FERMI
            elif arg.split(':')[1] == "modal":
                culture_update_rule = lib.UPDATE_RULE.UPDATE_RULE_MODAL
            elif arg.split(':')[1] == "modalbest":
                culture_update_rule = lib.UPDATE_RULE.UPDATE_RULE_MODALBEST
            else:
                sys.stderr.write('bad update rule ' + arg.split(':')[1]+'\n')
                sys.exit(1)

        elif (len(arg.split(':')) == 2 and
              arg.split(':')[0] == "init_strategy"):
            # initial strategy layout
            if arg.split(':')[1] == "random": # default
                init_strategy = lib.INIT_STRATEGY.RANDOM
            elif arg.split(':')[1] == "stripe":
                init_strategy = lib.INIT_STRATEGY.STRIPE
                print 'initial strategy stripe pattern'
            elif arg.split(':')[1] == "chessboard":
                init_strategy = lib.INIT_STRATEGY.CHESSBOARD
                print 'initial strategy chessboard pattern'
            else:
                sys.stderr.write('bad update rule ' + arg.split(':')[1]+'\n')
                sys.exit(1)
                
        elif (len(arg.split(':')) == 2 and
              arg.split(':')[0] == 'resumeFrom'):
              # resumeFrom:n,q,beta_p,beta_s,theta,init_random_prob,radius,num_joint_activities,pool_multiplier,noise,i
              # instead of running all parameters, start at the specified set
              # (only makes sense if all other parameters are same as run that
              # is being 'resumed' - start from the parameters MPI rank 0
              # was doing when job terminated. If this is specified then
              # results<rank>.csv  and parameter<rank>.csv files will be
              # appended not overwritten
              resargs = arg.split(':')[1].split(',')
              if (len(resargs) != 11):
                sys.stderr.write('expecting 11 values for resumeFrom:  resumeFrom:n,q,beta_p,beta_s,theta,init_random_prob,radius,num_joint_activities,pool_multiplier,noise,i\n')
                sys.exit(1)
              if resargs[5] == 'None':
                resinitrandomprob = None
              else:
                resinitrandomprob = float(resargs[5])
              resumeFrom = (int(resargs[0]), int(resargs[1]), 
                            float(resargs[2]), float(resargs[3]),
                            float(resargs[4]), resinitrandomprob,
                            int(resargs[6]),
                            int(resargs[7]), float(resargs[8]),
                            float(resargs[9]), 
                            int(resargs[10]))

        elif arg.isdigit():
            n_list.append(int(arg))


        else:
            cppmodel = arg
            if not os.path.isfile(cppmodel):
                print 'Model executable not found.'
                sys.exit()        

    if init_culture_csv_file != None:
        if len(n_list)  > 0:
            print 'Cannot provide n values for read_init_culture'
            sys.exit(1)
        if evolved_init_culture or evolved_init_culture2 or prototype_init_culture or trivial_ultrametric_init_culture or zero_init_culture:
            print 'Cannot have both read_init_culture:csvfile and evolved_init_culture'
            sys.exit(1)
        initial_culture = list()
        F = None
        q = 0
        for row in csv.reader(open(init_culture_csv_file)):
            initial_culture.append(array([int(x) for x in row]))
            if F == None:
                F = len(initial_culture[-1])
            else:
                if len(initial_culture[-1]) != F:
                    sys.stderr.write('bad row in ' + str(init_culture_csv_file)
                                     + ' expecting ' + str(F) + ' columns ' +
                                     ' but got ' + str(F) + '\n')
                    sys.exit(1)
            q = max(q, max(initial_culture[-1]) + 1) # +1 since values in 0..q-1
        n = len(initial_culture)
        if n > m**2:
            new_m = int(math.ceil(math.sqrt(n)))
            sys.stderr.write(('WARNING: n = %d agents but lattice only  ' +
                             '(m = %d)**2, m set to %d\n') %
                             (n, m, new_m))
            m = new_m
        n_list = [n]
        q_list = [q]
        
    elif n_list == None:
        print 'Please provide n values'
        sys.exit()

    if evolved_init_culture2:
        if ef2_r == None:
            ef2_r = F / 2
        else:
            if ef2_r > F:
                sys.stderr.write('value of r for evolved_init_culture2:r must be <= F (r = %d, F = %d)\n' % (ef2_r, F))
                sys.exit(1)
        
    if cppmodel == None:
        print 'Model executable not found.'
        sys.exit()          

    if runs < mpi_processes:
        sys.stderr.write('ERROR: have %d MPI tasks but only %d runs, cannot run as some tasks would have to start with run number > 0\n' % (mpi_processes, runs))
        sys.exit (1)

    scenario(scriptpath, runs, 'results/', tmax, F, m, False, False,
                                      beta_p_list, beta_s_list, 
                                      n_list, q_list, time_list, directedMigration, 
                                      cppmodel, end, evolved_init_culture,
                                      evolved_init_culture2,
                                      prototype_init_culture,
                                      trivial_ultrametric_init_culture,
                                      theta_list, initial_culture,
                                      init_culture_csv_file,
                                      init_random_prob_list,cluster_k, ef2_r,
                                      radius_list,
                                      noise_list,
                                      num_joint_activities_list,
                                      pool_multiplier_list,
                                      strategy_update_rule,
                                      culture_update_rule,
                                      resumeFrom,
                                      snapshot_time_list,
                                      zero_init_culture,
                                      init_strategy)

