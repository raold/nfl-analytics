#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "analysis/dissertation/figures/out"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

eval_season <- 2024
full_start <- 1999
seasons <- full_start:eval_season

exp_weight <- function(s, t, H) 0.5 ^ ((t - s) / H)

weights <- data.frame(
  season = rep(seasons, times = 3),
  H = factor(rep(c(3,4,5), each = length(seasons)))
)
weights$weight <- mapply(function(s, h) exp_weight(s, eval_season, as.numeric(as.character(h))),
                         weights$season, weights$H)

png(file.path(out_dir, "time_decay_weights.png"), width = 1000, height = 560)
par(mar = c(4.5, 4.5, 3, 1))
plot(seasons, weights$weight[weights$H==3], type = "l", lwd = 2, col = "#e41a1c",
     xlab = "Season", ylab = "Relative weight vs eval season", ylim = c(0,1),
     main = sprintf("Exponential decay weights centered on %d", eval_season))
lines(seasons, weights$weight[weights$H==4], lwd = 2, col = "#377eb8")
lines(seasons, weights$weight[weights$H==5], lwd = 2, col = "#4daf4a")
legend("topleft", legend = c("H=3", "H=4", "H=5"), lwd = 2,
       col = c("#e41a1c", "#377eb8", "#4daf4a"), bty = "n")
points(c(min(seasons), eval_season),
       c(exp_weight(min(seasons), eval_season, 3), exp_weight(eval_season, eval_season, 3)),
       pch = 19, col = "#e41a1c")
points(c(min(seasons), eval_season),
       c(exp_weight(min(seasons), eval_season, 4), exp_weight(eval_season, eval_season, 4)),
       pch = 19, col = "#377eb8")
points(c(min(seasons), eval_season),
       c(exp_weight(min(seasons), eval_season, 5), exp_weight(eval_season, eval_season, 5)),
       pch = 19, col = "#4daf4a")
dev.off()

message("Wrote ", file.path(out_dir, "time_decay_weights.png"))
