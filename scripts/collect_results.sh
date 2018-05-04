#!/bin/sh

# concatenate all the reults from each MPI task into one file (results.csv)
# with header as first line
# WARNING: clobbers results.csv
# ADS 25Oct2012

if [ $# -ne 0 ]; then
  echo "usage: $0" >&2
  exit 1
fi

OUTFILE=results.csv

echo 'n,m,F,phy_mob_a,beta_p,soc_mob_a,beta_s,r,s,tolerance,q,theta,init_random_prob,radius,noise_rate,num_joint_activities,pool_multiplier,run,time,avg_path_length,dia,avg_degree,cluster_coeff,corr_soc_phy,corr_soc_cul,corr_phy_cul,num_cultures,size_culture,overall_diversity,ass,num_components,largest_component,within_component_diversity,between_component_diversity,component_diversity_ratio,num_communities,largest_community,within_community_diversity,between_community_diversity,community_diversity_ratio,social_clustering,social_closeness,physical_closeness,overall_closeness,physicalStability,socialStability,culturalStability,num_culture_components,num_regions,max_region_size,num_cooperators,num_defectors,num_strategy_regions,max_strategy_region_size,avg_num_players,avg_num_cooperators,avg_did_cooperate,avg_cooperators_over_players,avg_did_cooperate_over_cooperators,avg_payoff,avg_defector_payoff,avg_cooperator_payoff,avg_mpcr,avg_pool_multiplier' > $OUTFILE

cat results/*/results?.csv results/*/results??.csv results/*/results???.csv >> $OUTFILE


