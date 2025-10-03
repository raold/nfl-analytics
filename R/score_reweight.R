#!/usr/bin/env Rscript
# Integer-margin reweighting to match key-number masses

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

#' Reweight integer margin pmf to match target masses
#' @param pmf tibble with columns margin (int), prob (numeric)
#' @param targets named numeric vector, e.g., c(`3`=0.09, `6`=0.06, `7`=0.07, `10`=0.05)
#' @return tibble margin/prob normalized
reweight_key_masses <- function(pmf, targets) {
  # Multiply probabilities at key margins by factor = target/current, then renormalize
  pmf2 <- pmf |>
    mutate(
      target = targets[as.character(margin)],
      factor = ifelse(!is.na(target) & prob > 0, target / prob, 1.0),
      prob = prob * factor
    ) |>
    select(-target, -factor)
  s <- sum(pmf2$prob)
  if (s <= 0) return(pmf2)
  pmf2 |>
    mutate(prob = prob / s)
}

if (identical(environment(), globalenv())) {
  demo <- tibble::tibble(margin = -10:10, prob = dpois(abs(-10:10), 3))
  demo <- demo |>
    mutate(prob = prob / sum(prob))
  targets <- c(`3` = 0.09, `6` = 0.06, `7` = 0.07, `10` = 0.05)
  out <- reweight_key_masses(demo, targets)
  print(head(out))
}
