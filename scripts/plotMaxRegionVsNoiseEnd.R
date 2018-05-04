#!/usr/bin/Rscript

# Plot noise on x axis and max cluster size on y axis for results
# from Axelrod / joint activity experiments. 
# this verision usies the output from collect_results.sh collecting results
# from the model e.g. /lattice-jointactivity-simcoop-social-noise-cpp-end/model
# run by
# lattice-python-mpi/src/axelrod/geo/expphysicstimeline/multiruninitmain.py
# but without 'end' on the command line so writes stats at various time points
# rather than running to equilibrium and writing only at end -
# however this script takes only the latest time point to plot results
# at end of time iterations (hopefully stochastic equilibrium).
# Also create spearate file with num clusters on y axis.
#
# The name of the CSV file with the data is given on stdin
#
# Usage:
#
# Rscript plotMaxRegionVsNoiseEnd.R data.csv outputprefix
#
#
# E.g. Rscript plotMaxRegionVsNoiseEnd.R results.csv varynoise-combined
#
# Output is output.eps files
# given on command line, with -max_region_size and -num_regions suffix  etc.
# e.g. outputprefix-max_region_size and outputprefix-num_regions.eps
# Also different output file for eacch value of num_joint_activities and radius
#
# If outputprefix containts 'constantmpcr' then the pool_multiplier id variable
# is actually the MPCR and output labels changed accordingly.
# If outputprefix contains 'final' then don't do facets and just have
# the hardcoded L and MPCR values in here for final version of plot for
# including in manuscript.
#
# ADS April 2016
#
# $Id: plotMaxRegionVsNoiseEnd.R 512 2016-04-29 05:15:15Z stivalaa $


library(ggplot2)
library(doBy)
library(reshape)
library(scales)
library(grid)

# http://stackoverflow.com/questions/1358003/tricks-to-manage-the-available-memory-in-an-r-session
# improved list of objects
.ls.objects <- function (pos = 1, pattern, order.by,
                        decreasing=FALSE, head=FALSE, n=5) {
    napply <- function(names, fn) sapply(names, function(x)
                                         fn(get(x, pos = pos)))
    names <- ls(pos = pos, pattern = pattern)
    obj.class <- napply(names, function(x) as.character(class(x))[1])
    obj.mode <- napply(names, mode)
    obj.type <- ifelse(is.na(obj.class), obj.mode, obj.class)
    obj.size <- napply(names, object.size)
    obj.dim <- t(napply(names, function(x)
                        as.numeric(dim(x))[1:2]))
    vec <- is.na(obj.dim)[, 1] & (obj.type != "function")
    obj.dim[vec, 1] <- napply(names, length)[vec]
    out <- data.frame(obj.type, obj.size, obj.dim)
    names(out) <- c("Type", "Size", "Rows", "Columns")
    if (!missing(order.by))
        out <- out[order(out[[order.by]], decreasing=decreasing), ]
    if (head)
        out <- head(out, n)
    out
}
# shorthand
lsos <- function(..., n=10) {
    .ls.objects(..., order.by="Size", decreasing=TRUE, head=TRUE, n=n)
}

bla <- function(variable, value) {

    # note use of bquote(), like a LISP backquote, evaluates only inside
    # .()
# http://stackoverflow.com/questions/4302367/concatenate-strings-and-expressions-in-a-plots-title
# but actually it didn't work, get wrong value for beta_s (2, even though no
   # such beta_s exists ?!? some problem with efaluation frame [couldn't get
# it to work by chaning where= arugment either), so using
# substitute () instead , which also doesn't work, gave up.
# --- turns out problem was I'd forgotten that all these vsariables have
# been converted to factors, so have to do .(levels(x)[x]) not just .(x)
    sapply      (value, FUN=function(xval ) 
        if (variable == "beta_s") {
          print(xval)
          bquote(beta[s]==.(levels(xval)[xval]))
        }
        else if (variable == "n") {
          bquote(N/L^2==.(levels(xval)[xval]))
        }
        else if (variable == "n_immutable") {
          bquote(F[I] == .(levels(xval)[xval]))
        }
        else if (variable == "m") {
          bquote(L == .(levels(xval)[xval]))
        }
        else if (variable == "pool_multiplier") {
          bquote(plain("pool multiplier") ~~ m == .(levels(xval)[xval]))
        }
        else if (variable == "mpcr") {
          bquote(plain("MPCR") == .(xval))
        }
        else {
          bquote(.(variable) == .(levels(xval)[xval]))
        }
      )
}


zSigma <- 1.96 # number of sd for 95% CI

Fvalue <- 5   # uses only one value of F

qs <- c(15, 35, 55, 65, 75, 85, 95)

responses <- c('max_region_size', 'num_regions', 'num_strategy_regions', 'max_strategy_region_size', 'num_cooperators','avg_num_players','avg_num_cooperators','avg_cooperators_over_players','avg_did_cooperate','avg_payoff','avg_defector_payoff','avg_cooperator_payoff','avg_mpcr','avg_pool_multiplier')
response_names <- c('Largest region size', 'Number of regions', 'Number of strategy regions', 'Largest strategy region size', 'Fraction of cooperators', 'Average number of players per game', 'Average number of cooperators per game', 'Averge fraction of cooperators per game', 'Averge number of cooperators that did cooperate per game', 'Average payoff', 'Average defector payoff', 'Average cooperator payoff', 'Average MPCR', 'Average pool multiplier')


if (length(commandArgs(trailingOnly=TRUE)) != 2) {
    cat("Usage: Rscript plotMaxRegionVsNoiseEnd.R results.csv outputprefix\n")
    quit(save='no')
}
results_filename <- commandArgs(trailingOnly=TRUE)[1]
base_output_prefix <- commandArgs(trailingOnly=TRUE)[2]

constant_mpcr <- FALSE
if (length(grep('constantmpcr', base_output_prefix)) > 0 ){
  constant_mpcr <- TRUE
}

final <- FALSE
if (length(grep('final', base_output_prefix)) > 0 ){
  final <- TRUE
}


orig_experiment <- read.table(results_filename, sep=',',header=T,stringsAsFactors=F)

end_time <- max(orig_experiment$time)
orig_experiment <- orig_experiment[which(orig_experiment$time == end_time),]
orig_experiment <- orig_experiment[which(orig_experiment$F == Fvalue),]
orig_experiment <- orig_experiment[which(orig_experiment$q %in% qs),]

if (final) {
     if (end_time == 1e08) {
         Lvalue <- 200
        orig_experiment <- orig_experiment[which(orig_experiment$m == Lvalue),]
        orig_experiment <- orig_experiment[which(orig_experiment$pool_multiplier == 0.6),]
     } else {
         Lvalue <- 100
        orig_experiment <- orig_experiment[which(orig_experiment$m == Lvalue),]
        orig_experiment <- orig_experiment[which(orig_experiment$pool_multiplier == 0.6),]
     }
}
experiment <- orig_experiment


num_joint_activities_list <- unique(experiment$num_joint_activities)
radius_list <- unique(experiment$radius)

for (njai in 1:length(num_joint_activities_list)) {
  for (radi in 1:length(radius_list)) {


  experiment <- orig_experiment


  experiment <- experiment[which(experiment$num_joint_activities == num_joint_activities_list[njai]),]


experiment <- experiment[which(experiment$radius == radius_list[radi]),] 
  output_prefix <- paste(base_output_prefix, '-num_joint_activities', num_joint_activities_list[njai], sep='')
  output_prefix <- paste(output_prefix, '-radius', radius_list[radi], sep='')

  experiment$m <- as.factor(experiment$m)
  if (constant_mpcr) {
    experiment$mpcr <- experiment$pool_multiplier
    experiment$pool_multiplier <- NA
    experiment <- experiment[which(experiment$mpcr < 1),]
  } else {
    experiment$pool_multiplier <- as.factor(experiment$pool_multiplier)
  }

  
# num_cooperators and num_defectorss is not normalized like others, so do it now
experiment$num_cooperators <- experiment$num_cooperators / experiment$n

if (constant_mpcr) {
  D <- melt(experiment, id=c('n','m','F','q','radius','mpcr','noise_rate','run','time') )
  D<-summaryBy(value ~ n + m + F + q + radius + mpcr + noise_rate + time + variable, data=D, FUN=c(mean, sd))
} else {
  D <- melt(experiment, id=c('n','m','F','q','radius','pool_multiplier','noise_rate','run','time') )
  D<-summaryBy(value ~ n + m + F + q + radius + pool_multiplier + noise_rate + time + variable, data=D, FUN=c(mean, sd))
}

  
for (i in 1:length(responses)) {
  response <- responses[i]
  if (!(response %in% colnames(experiment))) {
      print(paste('skipping response ', response, ' not in data'))
      next
  }
  response_name <- response_names[i]
  Dst <- D[which(D$variable == response), ]


  print('before ggplot'); print(lsos()) #XXX

  p <- ggplot(Dst, aes(x = noise_rate, y = value.mean,
                       colour = as.factor(q), linetype=as.factor(q), shape = as.factor(q))   )
  p <- p + theme_bw()
  p <- p + theme(plot.margin = 	    unit(c(0,0,0,0), "lines"),
		axis.text.x =       element_text(size = 12, colour = "black", lineheight = 0.2),
		axis.text.y =       element_text(size = 12, colour = "black", lineheight = 0.2),
		axis.title.x =       element_text(size = 12, colour = "black", lineheight = 0.2),
		axis.title.y =       element_text(angle = 90, size = 12, colour = "black", lineheight = 0.2),

	      strip.text.x =	element_text(size = 12, colour = "black"),
	      strip.text.y =	element_text(angle = -90, size = 10, colour = "black"),
        legend.text = 	element_text(size = 12, colour = "black"),
	      legend.key =  	element_rect(fill = "white", colour = "white"),

		axis.ticks =        element_line(colour = "black"),
		axis.ticks.length = unit(0.1, "cm"),
		strip.background =  element_rect(fill = "white", colour = "white"),
		panel.grid.minor =  element_blank(),
		panel.grid.major =  element_blank(),
		panel.border =      element_rect(colour = "black"),
		axis.ticks.margin = unit(0.1, "cm")
  )
  p <- p + theme(legend.position = c(1, 1), legend.justification = c(1, 1))
  p <- p + geom_point()
  p <- p + geom_errorbar(aes(ymin=value.mean - zSigma*value.sd, ymax=value.mean + zSigma*value.sd), width=0.1)
  p <- p + geom_line()
  p <- p + scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                        labels = trans_format("log10", math_format(10^.x)))

##   if (substr(response, 1, 3) != 'avg') {
## # for the moving averges e.g. avg_num_players, it is a number not normalized
##     p <- p + scale_y_continuous(limits = c(0, 1))
##   }
  p <- p + xlab('Noise rate')
  p <- p + ylab(response_name)
  if (final) {
      p <- p + ggtitle (bquote(list(L == .(Lvalue), F == .(Fvalue), plain(radius) == .(Dst$radius), plain(MPCR) == .(Dst$mpcr))))
  } else {
      if (constant_mpcr) {
          p <- p + facet_grid(mpcr ~ m, labeller=bla)
      } else {
          p <- p + facet_grid(pool_multiplier ~ m, labeller=bla)
      }
      p <- p + ggtitle (bquote(list(F == .(Fvalue), plain(radius) == .(Dst$radius))))
  }

  p <- p + scale_colour_brewer('q', palette = "Dark2")
  p <- p + scale_linetype('q')
  p <- p + scale_shape_manual('q', values=1:nlevels(as.factor(Dst$q))) # needed for more than 6 shapes

  print ('before postscript()   '); print(lsos()) #XXX

# EPS suitable for inserting into LaTeX
  postscript(paste(paste(output_prefix, response, sep='-'), 'eps',sep='.'),
             onefile=FALSE,paper="special",horizontal=FALSE, 
             width = 9, height = 6)

# Postscript fiels generate above are absurdly huge (840 MB +), so much
# so that even trying to view them with gs etc.
# takes longer than the simulation or plotting, so try using high res TIFF
# instead
# (Can't understand why this script generates such huge plots, while others,
# even with much larger amounts of data points, do not, or at least not as bad)
#tiff(paste(paste(output_prefix, response, sep='-'), 'tiff',sep='.'),
#     res = 300, width = 9, height = 6, units='in')
# doesn't work either as apparently "No TIFF support in this version of R"
# so have to use PNG instead
#png(paste(paste(output_prefix, response, sep='-'), 'png',sep='.'),
#     res = 300, width = 9, height = 6, units='in')

  print ('after postscript()   '); print(lsos()) #XXX

# crashes at the next line, too much memory; even 32GB on 64 bit system not enough
# No solution apparently:
# http://stackoverflow.com/questions/6850041/ggplot2-printing-plot-balloons-memory
  # but turns out commented out the 
# p <- p + annotation_logticks(sides="b")
# above fixes the problem, down to using only about 2GB now
  print( p)
  print ('after print(p) '); print(lsos()) #XXX
  dev.off()
  print ('after dev.off()'); print(lsos()) #XXX
}
}
}

