#!/usr/bin/Rscript

# Plot noise on x axis and max cluster size on y axis for results
# from Axelrod / joint activity experiments. 
# this verision usies the output from collect_results.sh collecting results
# from the model e.g. /lattice-jointactivity-simcoop-social-noise-cpp-end/model
# run by
# lattice-python-mpi/src/axelrod/geo/expphysicstimeline/multiruninitmain.py
# but without 'end' on the command line so writes stats at various time points
# rather than running to equilibrium and writing only at end.
#
#
# The name of the CSV file with the data is given on stdin
#
# Usage:
#
# Rscript plotMaxRegionVsTimeExamples.csv outputfilename.eps
#
#
# E.g. Rscript plotMaxRegionVsTimeExamples results.csv example_timeseries.eps
#
# Output is output.eps file given on command line.
#
# This script plots specific parameter time series as an example to
# put in the manuscript. It plots max region size and number of cooperators
# on the same plot.
#
# ADS April 2016


library(ggplot2)
library(doBy)
library(reshape)
library(scales)
library(RColorBrewer)
library(grid)
library(gridExtra)

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

# http://stackoverflow.com/questions/10762287/how-can-i-format-axis-labels-with-exponents-with-ggplot2-and-scales
scientific_10_orig <- function(x) {
  parse(text=gsub("e", " %*% 10^", scientific_format()(x)))
}

scientific_10 <- function(x) {
# also remove + and leading 0 in exponennt
  parse( text=gsub("e", " %*% 10^", gsub("e[+]0", "e", (x))) )
   
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
        else if (variable == "noise_rate") {
          parse(text=paste("r == ", gsub("e", " %*% 10^", levels(xval)[xval])))
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
mpcr_value <-  0.6  # uses only one value of pool multiplier (MPCR)

#ylimits=c(-0.25, 1.35)
ylimits=c(-0.1, 1)


if (length(commandArgs(trailingOnly=TRUE)) != 2) {
    cat("Usage: Rscript plotMaxRegionVstimeExample.R results.csv outputprefix\n")
    quit(save='no')
}
results_filename <- commandArgs(trailingOnly=TRUE)[1]
output_filename <- commandArgs(trailingOnly=TRUE)[2]


orig_experiment <- read.table(results_filename, sep=',',header=T,stringsAsFactors=F)

# put NA on time=0 moving average value as they are not valid then
# (for older model versions when 0 was writtn for this rather than NA)
for (moving_avg_name in c('avg_num_players','avg_num_cooperators','avg_cooperators_over_players','avg_did_cooperate')) {
  orig_experiment[which(orig_experiment$time == 0),moving_avg_name] <- NA
}

orig_experiment$mpcr <- orig_experiment$pool_multiplier
orig_experiment$pool_multiplier <- NA

qvalue_list <- c(15, 95)
noisevalue_list <- c(0, 1e-5)


radiusvalue <- 2
if (max(orig_experiment$time) == 1e08){
  Lvalue <- 200 
} else {
  Lvalue <- 100 
}

orig_experiment <- orig_experiment[which(orig_experiment$num_joint_activities == 5), ]
orig_experiment <- orig_experiment[which(orig_experiment$radius == radiusvalue ), ]
orig_experiment <- orig_experiment[which(orig_experiment$m == Lvalue), ] 
orig_experiment <- orig_experiment[which(orig_experiment$F == Fvalue),]
orig_experiment <- orig_experiment[which(orig_experiment$mpcr == mpcr_value),]

responses <- c("num_cooperators", "avg_num_players", "num_regions")
response_names <- c("Fraction of cooperators", "Avg. players per game", "Number of cultural regions")

orig_experiment$q <- factor(orig_experiment$q)
orig_experiment$noise_rate <- factor(orig_experiment$noise_rate)

experiment <- orig_experiment
print(nrow(experiment)) #XXX
experiment <- experiment[which(experiment$q %in% qvalue_list), ]
print(nrow(experiment)) #XXX
experiment <- experiment[which(experiment$noise_rate %in% noisevalue_list), ]
print(nrow(experiment)) #XXX

# num_cooperators and num_defectorss is not normalized like others, so do it now
experiment$num_cooperators <- experiment$num_cooperators / experiment$n
stopifnot(experiment$radius == 2)
neighbourhood_size <- 13 # von Neumann neihbourhood size for radius = 2
experiment$avg_num_players <- experiment$avg_num_players / neighbourhood_size

D <- melt(experiment, id=c('n','m','F','q','radius','mpcr','noise_rate','run','time') )
D<-summaryBy(value ~ n + m + F + q + radius + mpcr + noise_rate + time + variable, data=D, FUN=c(mean, sd))




print(nrow(D))#XXX
Dst <- D[which(D$variable %in% responses), ]
print(nrow(Dst))#XXX

# cannot find a better way to do this, why is everythign so difficult in R?
# I just want to put response_names rather than responses on legend,
# but unless we actually replace all the variable names, they come out in
# some wrong and apparently aribtrary order, making the plot completely wrong
Dst$varname <- NA
print(Dst) #XXX
for (i in 1:length(responses)) {
  Dst[which(Dst$variable == responses[i]),]$varname <- response_names[i]
}
print(Dst) #XXX


Dst$varname <- factor(Dst$varname)

print('before ggplot'); print(lsos()) #XXX

p <- ggplot(data = Dst, aes(x = time, y = value.mean, colour = varname, linetype = varname, shape = varname)) +
      geom_point() +
      geom_line() +
      geom_errorbar(aes(ymin=value.mean - zSigma*value.sd, ymax=value.mean + zSigma*value.sd))

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

  p <- p + theme(legend.position = "bottom")

p <- p + scale_colour_brewer(name="", palette="Dark2")
p <- p + scale_linetype(name="")
p <- p + scale_shape(name="")

p <- p + scale_y_continuous(limits=ylimits)
p <- p + scale_x_continuous(labels = scientific_10)
p <- p + xlab('Time (iterations)')
p <- p + ylab('')
  title_expr <- parse(text=paste("list(F ==", Fvalue, ", L == ", Lvalue, ", plain(MPCR) ==", mpcr_value, ", plain(radius) == ", radiusvalue, ")"))
#  p <- p + ggtitle (title_expr)


p <- p + facet_grid(q ~ noise_rate , labeller=bla)


# EPS suitable for inserting into LaTeX
postscript(output_filename,
           onefile=FALSE,paper="special",horizontal=FALSE, 
           width = 9, height = 6)
print(p)
dev.off()

