#!/usr/bin/env Rscript
# Gaussian copula dependence utilities

suppressPackageStartupMessages({
  library(stats)
  library(dplyr)
})

#' Fit Gaussian copula correlation from pseudo-observations
#' @param u numeric vector in (0,1)
#' @param v numeric vector in (0,1)
#' @return list(rho=correlation)
fit_gaussian_copula <- function(u, v) {
  stopifnot(length(u) == length(v))
  eps <- 1e-9
  z1 <- qnorm(pmin(1 - eps, pmax(eps, u)))
  z2 <- qnorm(pmin(1 - eps, pmax(eps, v)))
  rho <- suppressWarnings(cor(z1, z2, use = "complete.obs"))
  rho <- ifelse(is.finite(rho), max(-0.999, min(0.999, rho)), 0)
  list(rho = rho)
}

if (identical(environment(), globalenv())) {
  set.seed(42)
  u <- pnorm(rnorm(1000))
  v <- pnorm(0.3 * qnorm(u) + sqrt(1-0.3^2) * rnorm(1000))
  print(fit_gaussian_copula(u, v))
}

