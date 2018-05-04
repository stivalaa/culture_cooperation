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
 *
 *  $Id: model.cpp 620 2016-05-11 07:17:40Z stivalaa $
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <limits>
#include <cassert>
#include <unistd.h> // for getpid()


#include "model.hpp"

#define SVN_VERSION_MODEL_CPP "joint activity similarity based cooperation lattice social influence von Neumann model with noise constant MPCR rev 19"

// this version has the social network (graph) removed for
// efficiency on simpler model that doesnot use it

// modified by ADS to seed srand with time+pid and to also 
// ensure unique tmp files so safe for parallel execution with MPI Python
// NB this involved extra command line pameter to model to not compatible
// with unmodified versions
// Also added theta parameter as threshold for bounded confidence model

// This vesrion allows strategy rule update to be either Fermi function
// or best takes over

// This version varies the pool multiplier in the public goods game
// so that marginal per capita return (MPCR) is held constant rather
// than pool multiplier.
// Note that the pool_multiplier command line option used by multiruninitmain.py
// and lib.py has been changed to double and re-used as MPCR here rather
// than adding a new parameter.

const double fermi_K = 0.1; // uncertainly in Fermi function TODO make variable

static char *tempfileprefix = NULL;


static int num_cooperators = 0;
// moving averages
static double avg_num_players = 0; // number of players in the public goods game
static double avg_num_cooperators = 0; // number of playesr that are cooperators
static double avg_did_cooperate = 0;   // number of cooperates that did cooperate
static double avg_cooperators_over_players =0; // avg num_cooperators/num_players
static double avg_did_cooperate_over_cooperators =0; // avg did_cooperate/num_players
static double avg_payoff = 0;   // average payoff
static double avg_defector_payoff = 0; // average payoff for defectors
static double avg_cooperator_payoff = 0; // average payoff for cooperators
static double avg_mpcr = 0; // marginal per capita return
static double avg_pool_multiplier; // pool multiplier



// Read the time_list---the iterations at which the stats of the simulation are
// to be printed out---from a temporary file.
int read_time_list(unsigned long long int** time_list, int* n) {
        std::ifstream inFile ((toString(tempfileprefix) + ".T").c_str());

	if (inFile) {
		std::string line;

		// Get the first line and read out the number of time steps in the list
		if (!getline(inFile, line))
			return -1;

		*n = convert<unsigned long long int>(line);
		unsigned long long int* tmplist = new unsigned long long int[*n];

		// Get the list itself
		if (!getline(inFile, line))
			return -1;

		// Read every item of the list
		std::istringstream linestream(line);
		std::string item;
		int i = 0;
		while (getline (linestream, item, ',') and i < *n) {
			tmplist[i] = convert<unsigned long long int>(item);
			i++;
		}

		if (i != *n)
			return -1;

		*time_list = tmplist;
	} else {
		*n = 0;
	}

	return 0;
}


// Write the moving average statistics about the public goods games to filename
// in easy to read and parse format
//   avg_num_players = 1.12345
// etc.
void write_moving_averages(const char *filename,
                           double avg_num_players, 
                           double avg_num_cooperators, 
                           double avg_did_cooperate, 
                           double avg_cooperators_over_players,
                           double avg_did_cooperate_over_cooperators,
                           double avg_payoff,
                           double avg_defector_payoff,
                           double avg_cooperator_payoff,
                           double avg_mpcr,
                           double avg_pool_multiplier)
{
  std::ofstream outfile(filename);
  outfile << "avg_num_players = " << avg_num_players << std::endl;
  outfile << "avg_num_cooperators  = " << avg_num_cooperators << std::endl;
  outfile << "avg_did_cooperate = " << avg_did_cooperate << std::endl;
  outfile << "avg_cooperators_over_players = " << avg_cooperators_over_players << std::endl;
  outfile << "avg_did_cooperate_over_cooperators = " << avg_did_cooperate_over_cooperators << std::endl;
  outfile << "avg_payoff = " << avg_payoff << std::endl;
  outfile << "avg_defector_payoff = " << avg_defector_payoff << std::endl;
  outfile << "avg_cooperator_payoff = " << avg_cooperator_payoff << std::endl;
  outfile << "avg_mpcr = " << avg_mpcr << std::endl;
  outfile << "avg_pool_multiplier = " << avg_pool_multiplier << std::endl;
}


// Calculate Manhattan distance on the lattice between locatino (x1,y1) 
// and (x2,y2)
inline int manhattan_distance(int x1, int y1, int x2, int y2) {
  return abs(x2 - x1) + abs(y2 - y1);
}



// Calculate the cultural similarity between agents a and b as the proportion of
// features they have in common.
inline double similarity(Grid<int>& C, const int F, const int a, const int b) {
	int same = 0;
	for (int i = 0; i < F; i++)
			same += (C(a,i) == C(b,i));
	return (double)same/F;
}



void save(Grid<int>& lastC, Grid<double>& lastG, Grid<int>& C, Grid<double>& G, int n, int F) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			lastG(i,j) = G(i,j);
		}

		for (int j = 0; j < F; j++) {
			lastC(i,j) = C(i,j);
		}
	}
}


// test for equilbrium, return true if no more change is possible
bool stop2(Grid<int>& C, Grid<int>& L, int n, int F, double theta,
           int vonNeumann_radius) {
    // at equilibrium, all agents in a neighbourhood
    // must have either identical culture, or
    // or completely distinct culture (no traits in common, similarity = 0), or
    // cultures with similarity < theta.
    assert(theta >= 0.0 && theta <= 1.0);
    for (int i = 0; i < n; i++)  {
        for (int j = i+1; j < n; j++) { // symmetric, only need i,j where j > i
            if (manhattan_distance(L(i,0), L(i,1), L(j, 0), L(j,1)) > vonNeumann_radius) {
              // TODO this could be made more efficient by looping over only
              // agents in neighbourhood instead of this way of looping over
              // all and just doing next if not in neighbourhood.
              // But doesn't much matter in this versino since we only
              // use this when noise=0, in all other cases have to go
              // to iteration limit tmax anyway.
              continue;
            }
            double sim = similarity(C, F, i, j);
            assert(sim >= 0.0 && sim <= 1.0);
            // NB using >= and <= not == to for floating point comparison
            // with 0 or 1 since floating point == is dangerous, but
            // values cannot be < 0 or > 1, as just asserted so equivalent
            // to equality
            if ( !((sim >= 1.0 ) ||
                   (sim <= 0.0 || sim < theta )) ) {
                return false;
            }
        }
    }
    return true;
}


// 
// return list of feature indices (0..F-1) on which change is possible
// with respect to agent a, amonsgt the social influence list influence_list
// given the culture vectors C, num features F and num trait values q
// 
std::vector<int> possible_change_features(int a,
                                         std::vector<int>& influence_list, 
                                          Grid<int>& C, int F, int q)
{
  std::vector<int> feature_list;

  // change is possible on a feature if there is at least one trait 
  // that is shared by at least as many members of influence_list as 
  // the current value.
  for (int feature = 0; feature < F; feature++) {
    // get number of agenst in influence set with a's current value of feature
    int num_with_current_value = 0;
    for (std::vector<int>::iterator it = influence_list.begin();
          it != influence_list.end(); it++) {
      int b = *it;
      if (C(b, feature) == C(a, feature)) {
        num_with_current_value++;
      }
    }
    // now see if there is a trait value for feature,
    // which is shared by at least as many in influence set as the current
    // value
    std::vector<int> histogram(q, 0);
    for (int j = 0; j < (int) influence_list.size(); j++) {
      int b = influence_list[j];
      int qvalue = C(b, feature);
      assert(qvalue >=0 && qvalue < q);
      ++histogram[qvalue];
    }
    for (int i =0 ; i < (int)histogram.size(); i++) {
      if (histogram[i] >= num_with_current_value) {
        feature_list.push_back(feature);
      }
    }
  }
  return feature_list;
}

//
// get most frequent value of trait idx (0..F-1) among influence_list agents
// given culture vectors C, num geatures F, num trait values q
// if multiple traits are the mode then chose a random one, except
// if one is the current value current, then chose that.
// 
int get_modal_value(std::vector<int>& influence_list, int idx,  int current,
                    Grid<int>& C, int F, int q) {
  std::vector<int> histogram(q, 0);
  for (int i = 0; i < (int)influence_list.size(); i++) {
    int b = influence_list[i];
    int qvalue = C(b, idx);
    assert(qvalue >=0 && qvalue < q);
    ++histogram[qvalue];
  }
  int modal_freq = *(std::max_element(histogram.begin(), histogram.end()));
  std::vector<int> modal_values;
  for (int i = 0; i < q; i++) {
    if (histogram[i] == modal_freq) {
      modal_values.push_back(i);
      if (i == current) {
        return current; // use current trait value if it is one of the modes
      }
    }
  }
  assert(modal_values.size() >= 1);
  return modal_values[rand() % modal_values.size()];
}



//
// play the public goods game with the supplied list of agents
// Returns the list of payoffs for each agent (corresponding to agent_list)
// with constant marginal per capita return (MPCR), so pool multiplier
// varies with pool size to keep MPCR constant at supplied value.
// Cooperation (for cooperators) is not always, but proportional to cultural
// similarity to the focal agent
// TODO: this singling-out of the focal agent seems not the best, should
// perhaps be based on e.g. similarity to "prototype" (modal) culture vector
// amongst entire set of players?
//
void play_public_goods_game_constant_mpcr(std::vector<int>& agent_list,
                            Grid<int> & Strategy, /* Strategies n */
                            Grid<int> & C,    /* Culture values n x F */
                            int focal_agent,  /* focal agent */
                            int F,            /* number of traits */
                            double mpcr, /* marginal per capita return */
                            double payoffs[], /*OUT array size n */
                            double &avg_num_players, /* OUT moving average */
                            double &avg_num_cooperators, /* OUT moving average*/
                            double &avg_did_cooperate, /* OUT moving average */
                            double &avg_cooperators_over_players, /* OUT moving average */
                            double &avg_did_cooperate_over_cooperators, /* OUT moving average */
                            double &avg_payoff,  /* OUT moving average */
                            double &avg_defector_payoff, /* OUT moving avg */
                            double &avg_cooperator_payoff, /* OUT moving avg */
                            double &avg_mpcr, /* OUT moving avg */
                            double &avg_pool_multiplier) /* OUT moving avg */
{
  const int a = 1;                  // cost per agent to contribute 
  static long long num_games = 0;   // count number of times this func called
  int       pool = 0;               // value in public goods game pool
  int       n = agent_list.size();  // total number in game
  int       n_cooperate = 0;        // number of cooperators
  int       n_did_cooperate = 0;    // number of cooperators that did cooperate
  bool did_cooperate[agent_list.size()]; // did this agent cooperate?

//  std::cerr << "xxx " << "n = " << n << " r = " << r << std::endl;

  num_games++;

  for (int i = 0; i < n; i++) {
    did_cooperate[i] = false;
  }

  double r = mpcr * n; // pool multiplier = MPCR * group size
  for (int i = 0; i < n; i++) {
    assert(Strategy(agent_list[i], 0) == STRATEGY_COOPERATE || 
           Strategy(agent_list[i], 0) == STRATEGY_DEFECT);
    // co-operators only co-operate with probabiliyt of cultural similarity
    // to focal agent (see comment above)
    if (Strategy(agent_list[i], 0) == STRATEGY_COOPERATE) {
      n_cooperate += 1;
      if ( rand()/(float)RAND_MAX < similarity(C, F, focal_agent, agent_list[i])) {
        pool += r*a;
        did_cooperate[i] = true;
        n_did_cooperate++;
      }
    }
  }

  double sum_payoffs = 0;
  double sum_defector_payoffs = 0;
  double sum_cooperator_payoffs = 0;
  for (int i = 0; i < n; i++) {
    int cost = did_cooperate[i] ? a : 0;
    double p = (double)pool/n - cost;
    payoffs[i] = p;
    sum_payoffs += p;
    sum_defector_payoffs += did_cooperate[i] ? 0 : p;
    sum_cooperator_payoffs += did_cooperate[i] ? p : 0;
  }
  double avg_payoff_this_game = sum_payoffs / n;
  double avg_defector_payoff_this_game = sum_defector_payoffs / n;
  double avg_cooperator_payoff_this_game = sum_cooperator_payoffs / n;

  double cooperators_over_players = (double)n_cooperate / n;
  // This can be a division by zero (n_cooperate can be 0)
  // but at least on the compilers/systems
  // I'm using it just gives NaN which is fine:
  double did_cooperate_over_cooperators = (double)n_did_cooperate / n_cooperate;


  // update moving averages: number of players, number of cooperators,
  // number of cooperates which did cooperate, etc.
  avg_num_players += (n - avg_num_players) / num_games;
  avg_num_cooperators += (n_cooperate  - avg_num_cooperators) / num_games;
  avg_did_cooperate += (n_did_cooperate - avg_did_cooperate) / num_games;
  avg_cooperators_over_players += (cooperators_over_players - avg_cooperators_over_players)  / num_games;
  avg_did_cooperate_over_cooperators += (did_cooperate_over_cooperators - avg_did_cooperate_over_cooperators) / n;
  avg_payoff += (avg_payoff_this_game - avg_payoff) / num_games;
  avg_defector_payoff += (avg_defector_payoff_this_game - avg_defector_payoff) / num_games;
  avg_cooperator_payoff += (avg_cooperator_payoff_this_game - avg_cooperator_payoff) / num_games;
  avg_mpcr += (mpcr - avg_mpcr) / num_games;
  avg_pool_multiplier += (r - avg_pool_multiplier) / num_games;
  

//  std::cout << "n = " << n << " n_cooperate = " << n_cooperate << " n_did_cooperate = " <<  n_did_cooperate << std::endl; //XXX
//  std::cout << "avg_num_players = " << avg_num_players << " avg_num_cooperators = " << avg_num_cooperators << " avg_did_cooperate " << avg_did_cooperate << std::endl; //XXX
//  std::cout << "avg_cooperators_over_players = " << avg_cooperators_over_players << " avg_did_cooperate_over_cooperators = " << avg_did_cooperate_over_cooperators << std::endl;

#ifdef DEBUG
  std::cout << "aaa payoffs: ";
  for (int i = 0; i < n; i++)
    std::cout << payoffs[i] << " ";
  std::cout << std::endl;
#endif
}


// used for max_element to compare values
bool payoffpaircompare(const std::pair<int, double>& p1, 
                       const std::pair<int, double>& p2) {
  return p1.second < p2.second; 
}

//
// given map of agentid to total payoff,
// return vector of agents ids with equal highest payoff 
//
std::vector<int> get_equal_best_payoff_agents(std::unordered_map<int, double>& payoffs)
{
#ifdef DEBUG
  std::cerr << "xxx payoffs: "  ;
  for (std::unordered_map<int, double>::iterator it = payoffs.begin();
       it != payoffs.end(); it++) {
    std::cerr << it->first << " = " << it->second << ",  " ;
  }
  std::cerr << std::endl;
#endif
  double max_payoff = (std::max_element(payoffs.begin(), payoffs.end(),
                                        payoffpaircompare))->second;
//  std::cerr << "xxx" << "max_payoff = " << max_payoff << std::endl;
  std::vector<int> max_payoff_agents;
  for (std::unordered_map<int, double>::iterator it = payoffs.begin();
       it != payoffs.end(); it++) {
    if (it->second >= max_payoff) { // use >= not == on double 
      max_payoff_agents.push_back(it->first);
    }
  }
  assert(max_payoff_agents.size() >= 1);
  return max_payoff_agents;
}

//
// given map of agentid to total payoff,
// return agent id with best payoff, breaking ties randomly if more than one
//
int best_payoff_agent(std::unordered_map<int, double>& payoffs)
{
  std::vector<int> max_payoff_agents = get_equal_best_payoff_agents(payoffs);
  return max_payoff_agents[rand() % max_payoff_agents.size()];
}


// 
// Fermi function
// given two payoffs p1,p2, uncertainty constant K (e.g. 0.1 frequently used)
// and number of players in the reelvant group G, compute the Fermi function
// for probabiilty of updating player strategy.
// Here p1 is the payoff of the player onto whom the update may be forced 
// (i.e. when p1 < p2 we want higher probability)
//
inline double fermi(double p1, double p2, double K) {
  double delta = p1 - p2;
  return 1.0 / (1.0 + exp(delta/K));
}

//
// given map of agentid to total payoff,
// return modal strategy among the agents with the highest payoff
// if equal numbers, use the focal agent's current value
int modal_strategy_best_payoff_agents(const Grid<int>& Strategy,
                                      int current_strategy,
                                      std::unordered_map<int, double>& payoffs)
{
  int modal_strategy = current_strategy;

  std::vector<int> max_payoff_agents = get_equal_best_payoff_agents(payoffs);
  int cooperator_count = 0;
  for (std::vector<int>::iterator it = max_payoff_agents.begin();
       it != max_payoff_agents.end(); it++) {
    cooperator_count += (Strategy(*it, 0) == STRATEGY_COOPERATE);
  }
  int defector_count = max_payoff_agents.size() - cooperator_count;
  if (defector_count == cooperator_count) {
    modal_strategy = current_strategy;
  }
  else if (defector_count > cooperator_count) {
    modal_strategy = STRATEGY_DEFECT;
  }
  else {
    modal_strategy = STRATEGY_COOPERATE;
  }
  return modal_strategy;
}

//
// the model main loop
// 
unsigned long long model(Grid<int>& L, Grid<int>& O, Grid<int>& C,
                         Grid<int>& Strategy,
                         unsigned long long int tmax, int n, int m, int F,
		int q, double r, double s,
		bool toroidal, bool network, double tolerance, bool directed_migration,
		double phy_mob_a, double phy_mob_b, double soc_mob_a, double soc_mob_b,
		double r_2, double s_2, double phy_mob_a_2, double phy_mob_b_2, double soc_mob_a_2, double soc_mob_b_2,
		double k, unsigned long long int timesteps[], long int time_list_length, std::ofstream& log, 
                         double theta, int vonNeumann_radius, double noise_level,
                         int num_joint_activities, double mpcr,
                         UpdateRule_e strategy_update_rule,
                         UpdateRule_e culture_update_rule
  ) {
       srand(time(NULL)+(unsigned int)getpid()); // so processes started at same time have different seeds (time is only second resolution)



	std::cout << "tmax: " << tmax << std::endl;
	std::cout << "n: " << n << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "F: " << F << std::endl;
	std::cout << "q: " << q << std::endl;


    double w[n];

    double sumw = 0.0;


	double nw[n];
	double cw[m*m];

    int a, b, idx;
    int nextstep = 0;

  // at the moment always have r = r' as in Flache & Macy (2011)
  double selection_noise_level = noise_level;   // r' in Flache & Macy (2011)
  double interaction_noise_level = noise_level; // r in Flache & Macy (2011)

int best;


	// run model
    for (unsigned long long t = 0; t < tmax; t++) {
    	// If this iteration is in the time list, write out the current state
    	// of the simulation.
    	if (nextstep < time_list_length and timesteps[nextstep] == t) {
	  L.write((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".L").c_str(), ',');
	  C.write((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".C").c_str(), ',');
    Strategy.write((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".Strategy").c_str(), ',');
    write_moving_averages((toString(tempfileprefix) + "-" + toString(timesteps[nextstep]) + ".Gamestats").c_str(), avg_num_players, avg_num_cooperators, avg_did_cooperate, avg_cooperators_over_players, avg_did_cooperate_over_cooperators, avg_payoff, avg_defector_payoff, avg_cooperator_payoff, avg_mpcr, avg_pool_multiplier);
    		nextstep++;
//     std::cout << "t = " << t << ", num_cooperators = " << num_cooperators << std::endl;//XXX
#undef DEBUG2
#ifdef DEBUG2
     int nc = 0;
     for (int i = 0; i < n; i++) {
       if (Strategy(i, 0) == STRATEGY_COOPERATE) {
        nc++;
       }
     }
     std::cerr << "t = " << t << ", num_cooperators = " << num_cooperators << " nc = " << nc << std::endl;
       assert(nc == num_cooperators);
#endif /*DEBUG2*/
    	}
      

    	if (t == 50000 || t == 100000 || t == 500000 || t == 1000000 || (t > 0 && t % 10000000 == 0)) {
    		std::cout << "Reaching " << t << " iterations." << std::endl;
//			save(lastC, lastG, C, G, n, F);
        // Only check absorbing state if there is no noise and not doing time series
			  if (noise_level == 0.0 && time_list_length == 0 && stop2(C, L, n, F, theta, vonNeumann_radius)) {
  		  		std::cout << "Stopping after " << t << " iterations." << std::endl;
            return t;
  			}
      }


    	// Draw one agent randomly.
    	a = rand() % n;

        // Make lsit of neigbhours in von Neumann
        // neighbourhood of the focal agent.
        int ax = L(a,0);
        int ay = L(a,1);
        std::vector<int>neighbours;
        for (int xi = -1*vonNeumann_radius; xi <= vonNeumann_radius; xi++) {
            for (int yi = -1*vonNeumann_radius; yi <= vonNeumann_radius; yi++) {
                if ((xi != 0 || yi != 0) &&  // don't include focal agent itself
                    abs(xi) + abs(yi) <= vonNeumann_radius) {
                    int bx = ax + xi;
                    int by = ay + yi;
                    // handle edge of lattice, not toroidal
                    if (bx >= 0 && bx < m && by >= 0 && by < m) {
                        neighbours.push_back(O(bx, by));
                    }
                }
            }
        }
     
        // becuase we have sveral different subsets of players in different
        // joint activities, we will maintain a hash map mapping agent id
        // to payoff for al lpotential players
        std::unordered_map<int, double> payoff_map;
        // also (for modal) need list of players, but no C++ STL method
        // to get vector of keys from unordered_map so have to do it manually
        std::vector<int>all_other_players; // does not include focal agent

        // play each of the num_joint_acivities public goods games
        for (int j = 0; j < num_joint_activities; j++) {
          // for each game we apply the homophily rule in the neighbourhood
          // to get potentially different subsets of players in each of the games
          // social influence: consider all neighbours

          std::vector<int>influence_list; // neighbours with sufficient similarity
          for (std::vector<int>::iterator it = neighbours.begin();
               it != neighbours.end(); it++) {
            b = *it;
            // With the probability of their attraction,
            // a and b can interact successfully.
            double sim = similarity(C, F, a, b);
            // "bounded confidence" with theta as threshold:
            // unsuccesful interaction if similarity is less than theta
            bool do_interaction = false;
            if (sim >= theta) {
              if (rand()/(float)RAND_MAX < sim) {
                do_interaction = true;
              }
              // introduce selection error (as per setep 3 in Flache & Macy (2011))
              // the interaction decision is reversed with probabililty
              // selection_noise_level (r')
              if (rand()/(float)RAND_MAX < selection_noise_level) {
                do_interaction = !do_interaction;
              }
            }
            if (do_interaction)  {
              influence_list.push_back(b);
            }
          }
          
          if (influence_list.size() > 0) {
            std::vector<int> players(influence_list);
            players.push_back(a); // add focal agent as a player on end
            int num_players = players.size();
            double payoffs[num_players];
            play_public_goods_game_constant_mpcr(players, Strategy, C, a, F, mpcr,
                                   payoffs, avg_num_players, 
                                   avg_num_cooperators, avg_did_cooperate,
                                   avg_cooperators_over_players,
                                   avg_did_cooperate_over_cooperators,
                                   avg_payoff,
                                   avg_defector_payoff,
                                   avg_cooperator_payoff,
                                   avg_mpcr,
                                   avg_pool_multiplier);
            for (int i = 0; i < (int) players.size(); i++) {
              if (payoff_map.find(players[i]) != payoff_map.end())  {
                payoff_map[players[i]] += payoffs[i];
              }
              else {
                payoff_map[players[i]] = payoffs[i]; 
                if (players[i] != a) { // do not include focal agent
                  all_other_players.push_back(players[i]);
                }
              }
            }
          }
        }
        
        
#ifdef DEBUG3
  std::cerr << "payoff_map.size() == " << payoff_map.size() << " all_other_players.size() == " << all_other_players.size() << std::endl;
  std::cerr << "zxxx payoff_map (focal agent = " << a << ")"  ;
  for (std::unordered_map<int, double>::iterator it = payoff_map.begin();
       it != payoff_map.end(); it++) {
    std::cerr << it->first << " = " << it->second << ",  " ;
  }
  std::cerr << std::endl;
  std::cerr << "all_other_players: ";
  for (std::vector<int>::iterator it = all_other_players.begin();
       it != all_other_players.end(); it++) {
    std::cerr << *it << ", ";
  }
  std::cerr << std::endl;
#endif //DEBUG3

        // update strategy  and culture vector
        if (payoff_map.size() > 1) {
          assert(all_other_players.size() == payoff_map.size() -1);
          int old_strategy = Strategy(a,0);
          int new_strategy = old_strategy;
          std::unordered_map<int, double>::iterator it;
          double prob, a_payoff, b_payoff;
          std::vector<int>features;
          std::vector<int>max_payoff_agents;

//      std::cout << "t = " << t << " best payoff agent = " << best_payoff_agent(payoff_map) << " strategy = " << Strategy(best_payoff_agent(payoff_map),0) << std::endl; //XXX

          // update strategy
          switch (strategy_update_rule) {
            case UPDATE_RULE_BESTTAKESOVER:
              // best takes over
              new_strategy = Strategy(best_payoff_agent(payoff_map), 0);
              break;

            case UPDATE_RULE_FERMI:
              // choose a random player and update focal agent with 
              // that player's strategy with probability from Fermi function
              // of differentce between their payoffs
              a_payoff = payoff_map.find(a)->second;
              assert(all_other_players.size() > 0);
              b = all_other_players[rand() % all_other_players.size()];
              assert(a != b);
              b_payoff = payoff_map.find(b)->second;
#ifdef DEBUG3
              std::cerr << "a = " << a << " b = " << b << std::endl;
              std::cerr << "a_payoff = " << a_payoff << " b_payoff = " << b_payoff << std::endl;
#endif // DEBUG3
              prob = fermi(a_payoff, b_payoff, fermi_K);
//              std::cout << "fermi prob = " << prob << std::endl;//XXX
              if (rand()/(float)RAND_MAX < prob) {
                new_strategy = Strategy(b, 0);
              }
              break;

           case UPDATE_RULE_MODAL:
              std::cerr << "modal update not valid on strategy" << std::endl;
              assert(false);
              break;

           case UPDATE_RULE_MODALBEST:
               // modal strategy among equal best payoff agents
               new_strategy = modal_strategy_best_payoff_agents(Strategy,
                                                                old_strategy,
                                                                payoff_map);
               break;

           default:
              std::cerr << "bad update rule" << strategy_update_rule << std::endl;
              assert(false);
              break;
          }
          Strategy(a,0) = new_strategy;
          if (old_strategy != Strategy(a,0)) {
            if (Strategy(a,0) == STRATEGY_COOPERATE) {
              num_cooperators++;
            }
            else {
              num_cooperators--;
            }
          }
          
          // update culture according to either "best takes over" or
          // Fermi rule a random feature is chosen and the focal agent
          // adopts the trait value on that feature
          // or the modal trait update as used in Flache & Macy (2011)

          switch (culture_update_rule) {
            case UPDATE_RULE_BESTTAKESOVER:
              // get trait from the agent with the highest payoff (with
              // uniform random tie breaking).
              best = best_payoff_agent(payoff_map);
              if (similarity(C, F, a, best) < 1.0) {
                do {
                  idx = rand() % F;
                } while (C(a, idx) == C(best, idx));
                C(a, idx) = C(best, idx);
              }
              break;

            case UPDATE_RULE_FERMI:
              // choose a random player and update focal agent with 
              // trait from that palyer with probability from Fermi function
              // of differentce between their payoffs
              a_payoff = payoff_map.find(a)->second;
              assert(all_other_players.size() > 0);
              b = all_other_players[rand() % all_other_players.size()];
              assert(a != b);
              b_payoff = payoff_map.find(b)->second;
              prob = fermi(a_payoff, b_payoff, fermi_K);
//              std::cout << "fermi prob culture = " << prob << std::endl;//XXX
              if (rand()/(float)RAND_MAX < prob) {
                if (similarity(C, F, a, b) < 1.0) {
                  do {
                    idx = rand() % F;
                  } while (C(a, idx) == C(b, idx));
                  C(a, idx) = C(b, idx);
                }
              }
              break;

            case UPDATE_RULE_MODAL:
              // modal trait value on feature where change is possible,
              // as per Flache & Macy (2011)
              // Get list of features on which change is possible
              if (all_other_players.size() > 0) {
                features = possible_change_features(a, all_other_players,
                                                                    C, F, q);
                // change one random feature on a to modal value in influence set
                if (features.size() > 0) {
                    idx = features[rand() % features.size()];
                    int modal_value = get_modal_value(all_other_players, idx, C(a, idx),
                                                      C, F, q);
                    C(a,idx) = modal_value;

                }
              }
              break;

            case UPDATE_RULE_MODALBEST:
              // modal trait value on feature where change is possible
              // among the equal best payoff agents
              max_payoff_agents = get_equal_best_payoff_agents(payoff_map);
              features = possible_change_features(a, max_payoff_agents, C, F, q);
              // change one random feature on a to modal value in equal best payoff set
              if (features.size() > 0) {
                  idx = features[rand() % features.size()];
                  int modal_value = get_modal_value(max_payoff_agents, idx, C(a, idx),
                                                    C, F, q);
                  C(a,idx) = modal_value;

              }
              break;

            default:
              std::cerr << "bad culture update rule" << culture_update_rule << std::endl;
              assert(false);
              break;
          }

        }

        // as in KETS step 6 in Flache & Macy (2011) interaction noise
        // is introduced by changing a random trait to a random value
        // with probability interaction_noise_level (r)
        if (rand()/(float)RAND_MAX < interaction_noise_level) {
          idx = rand() % F;
          C(a, idx) = rand() % q;
        }
    }
    std::cout << "Stopping after tmax = " << tmax << " iterations." << std::endl;
    return tmax;
}




int main(int argc, char* argv[]) {
	std::ofstream log("log.txt");

	// If the binary file is called with the argument -v, only the svn version
	// this binary was compiled from is printed.
	if (argc == 2 and argv[1][0] == '-' and argv[1][1] == 'v') {
		std::cout << "model.hpp: " << SVN_VERSION_MODEL_HPP << ", model.cpp: " << SVN_VERSION_MODEL_CPP << std::endl;
		return 0;
	}


	// Otherwise set default model arguments.
	int n = 100, m = 10, F = 5, q = 15;

	unsigned long long int tmax = 100000;
	double r = 1;
	double s = 1;

	bool toroidal = false;
	bool network = false;

    double tolerance = -1;
    bool directed_migration = false;

    double phy_mob_a = 1;
    double phy_mob_b = 10;
    double soc_mob_a = 1;
    double soc_mob_b = 10;

	double r_2 = r;
	double s_2 = s;

    double phy_mob_a_2 = phy_mob_a;
    double phy_mob_b_2 = phy_mob_b;
    double soc_mob_a_2 = soc_mob_a;
    double soc_mob_b_2 = soc_mob_b;

    double k = 0.01;

   double theta = 0.0;
    // size of the von Neumann neighbourhood for actors to interact, i.e.
    // the maximum Manhattan distance between interacting actors
    int vonNeumann_radius = 1;

    double noise_level = 0.0; // rate of noise

    int num_joint_activities = 10; // number of different public goods games
    double mpcr = 1; // marginal per capita return in public goods games

    UpdateRule_e strategy_update_rule = UPDATE_RULE_BESTTAKESOVER;
    UpdateRule_e culture_update_rule = UPDATE_RULE_BESTTAKESOVER;
    

    // If there are arguments, assume they hold model arguments in the following
    // order.
	if (argc > 1) {
		int index = 1;
		tmax = atoll(argv[index++]);
		n = atoi(argv[index++]);
		m = atoi(argv[index++]);
		F = atoi(argv[index++]);
		q = atoi(argv[index++]);
		r = atof(argv[index++]);                     // not used
		s = atof(argv[index++]);                     // not used
		toroidal = atoi(argv[index++]);              // not used
		network = atoi(argv[index++]);               // not used
		tolerance = atof(argv[index++]);             // not used
		directed_migration = atoi(argv[index++]);    // not used
		phy_mob_a = atof(argv[index++]);             // not used
		phy_mob_b = atof(argv[index++]);             // not used
		soc_mob_a = atof(argv[index++]);             // not used
		soc_mob_b = atof(argv[index++]);             // not used

		r_2 = atof(argv[index++]);                   // not used
		s_2 = atof(argv[index++]);                   // not used
		phy_mob_a_2 = atof(argv[index++]);           // not used
		phy_mob_b_2 = atof(argv[index++]);           // not used
		soc_mob_a_2 = atof(argv[index++]);           // not used
		soc_mob_b_2 = atof(argv[index++]);           // not used
		k = atof(argv[index++]);                     // not used
		tempfileprefix = argv[index++];
    theta = atof(argv[index++]);
    vonNeumann_radius = atoi(argv[index++]);   // von Neumann radius
    noise_level = atof(argv[index++]);         // noise level
    num_joint_activities = atoi(argv[index++]);    // number of games
    mpcr = atof(argv[index++]); // marginal per capita return
    strategy_update_rule = (UpdateRule_e)atoi(argv[index++]); // strategy update rule
    culture_update_rule = (UpdateRule_e)atoi(argv[index++]); // culture vector update rule
	}


    if (toroidal) {
    	std::cout << "alarm, toroidal not supported at the moment" << std::endl;
    	return -1;
    }

   if (n != m*m) {
    std::cerr << "for von Neumann neighbourrhood model must have n =m*m so all latice points used, but m = " << m << " and n = " << n <<std::endl;
    return -1;
   }
	// Try to read the list of iterations that determine when statistics are to
	// be created from a temporary file.
	unsigned long long int* time_list = NULL;
	int time_list_length = 0;
	int res = read_time_list(&time_list, &time_list_length);
	if (res == -1) {
		std::cout << "The time list file could not be read or there was a problem with its format." << std::endl;
		return -1;
	}

	Grid<int> L(n,2,-1);
	Grid<int> C(n,F,0);
	Grid <int> O(m,m,-1); // in this version O(x,y) contains agent id at that location or -1 if unoccupied


        // joint activity (public goods game) variables
        Grid<int>Strategy(n,1,STRATEGY_COOPERATE);

	if (argc == 1) {
		std::cerr << "must provide initialization parameters" << std::endl;
	} else {
		// load data from file
          //G.read((toString(tempfileprefix) + ".adj").c_str(), ' ');
 	        L.read((toString(tempfileprefix) + ".L").c_str(), ',');
	        C.read((toString(tempfileprefix) + ".C").c_str(), ',');
          Strategy.read((toString(tempfileprefix) + ".Strategy").c_str(),
                              ',');
	}

	for (int i = 0; i < n; i++)
		O(L(i,0), L(i,1)) = i; // O(x,y) is agent at lattice location (x,y)

  for (int i =0; i < n; i++)
    if (Strategy(i,0) == STRATEGY_COOPERATE)
      num_cooperators++;

	// Call the model
        unsigned long long tend = model(L, O, C, Strategy, tmax, n, m, F, q, r, s, toroidal, network, tolerance, directed_migration,
    		phy_mob_a, phy_mob_b, soc_mob_a, soc_mob_b,
    		r_2, s_2, phy_mob_a_2, phy_mob_b_2, soc_mob_a_2, soc_mob_b_2, k,
    		time_list, time_list_length, log, theta, vonNeumann_radius,
                noise_level, num_joint_activities, mpcr,
                strategy_update_rule, culture_update_rule);

        std::cout << "Last iteration: " << tend << std::endl;
    
        // Write out the state of the simulation
        L.write((toString(tempfileprefix) + ".L").c_str(), ',');
        C.write((toString(tempfileprefix) + ".C").c_str(), ',');
        Strategy.write((toString(tempfileprefix) + ".Strategy").c_str(), ',');
        write_moving_averages((toString(tempfileprefix) + ".Gamestats").c_str(),
                              avg_num_players, avg_num_cooperators, 
                              avg_did_cooperate, avg_cooperators_over_players,
                              avg_did_cooperate_over_cooperators,
                              avg_payoff, avg_defector_payoff,
                              avg_cooperator_payoff, avg_mpcr,
                              avg_pool_multiplier);

	std::ofstream outFile((toString(tempfileprefix) + ".Tend").c_str());
	outFile << tend;


	delete[] time_list;
	time_list = NULL;

	std::cout << "Fin" << std::endl;
	return 0;
}
