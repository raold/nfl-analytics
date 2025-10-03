#!/usr/bin/env Rscript
# Informative placeholder figure generator (PNG), base R only

open_dev <- function(path, w = 1200, h = 700, res = 150, mar = c(4, 5, 3, 2)) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  if (file.exists(path)) return(NULL)
  png(filename = path, width = w, height = h, res = res)
  op <- par(mar = mar, bg = "white")
  return(op)
}

close_dev <- function(op) {
  if (!is.null(op)) par(op)
  dev.off()
}

draw_note <- function(text_line = "Placeholder: replace via Quarto notebooks") {
  mtext(text_line, side = 3, line = 0.2, cex = 0.8, col = "#666666")
}

# 1) Time-decay weights (H in {3,4,5})
plot_time_decay <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  seasons <- 1999:2024; t <- 2024; Hs <- c(3,4,5)
  cols <- c("#1f77b4", "#ff7f0e", "#2ca02c")
  plot(NA, xlim = range(seasons), ylim = c(0,1), xlab = "Season", ylab = "Relative weight", main = "Time-Decay Weights (centered on 2024)")
  for (i in seq_along(Hs)) {
    H <- Hs[i]; w <- 0.5 ^ ((t - seasons)/H)
    lines(seasons, w, lwd = 2, col = cols[i])
  }
  legend("topright", legend = paste0("H=", Hs), col = cols, lwd = 2, bty = "n")
  draw_note()
  invisible(TRUE)
}

# 2) Rolling metrics: simple 4 blocks, two series
plot_rolling_metric <- function(path, ylab, main) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  blocks <- c("2011–14","2015–18","2019–21","2022–24")
  x <- seq_along(blocks)
  set.seed(42)
  recent <- if (ylab == "Log loss") c(0.695, 0.690, 0.700, 0.705) else c(0.045,0.043,0.049,0.052)
  decayed <- if (ylab == "Log loss") recent - c(0.006,0.004,0.005,0.004) else pmax(0, recent - c(0.004,0.003,0.004,0.003))
  ylim <- range(c(recent, decayed))
  plot(NA, xlim = range(x), ylim = ylim, xaxt = "n", xlab = "Evaluation block", ylab = ylab, main = main)
  axis(1, at = x, labels = blocks)
  lines(x, recent, lwd = 2, col = "#1f77b4"); points(x, recent, pch = 16, col = "#1f77b4")
  lines(x, decayed, lwd = 2, col = "#ff7f0e"); points(x, decayed, pch = 16, col = "#ff7f0e")
  legend("topleft", c("recent-only","decayed-full"), col = c("#1f77b4","#ff7f0e"), lwd = 2, bty = "n")
  draw_note()
}

# 3) Reliability curves: 45-degree line + two curves
plot_reliability <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  p_bins <- seq(0.05, 0.95, by = 0.10)
  obs_recent <- p_bins + (sin(seq_along(p_bins))*0.02)
  obs_decayed <- p_bins + (cos(seq_along(p_bins))*0.01)
  plot(NA, xlim = c(0,1), ylim = c(0,1), xlab = "Predicted", ylab = "Observed", main = "Reliability Curves (2024)")
  abline(0,1, col = "#888888", lty = 2)
  lines(p_bins, obs_recent, col = "#1f77b4", lwd = 2); points(p_bins, obs_recent, pch = 16, col = "#1f77b4")
  lines(p_bins, obs_decayed, col = "#ff7f0e", lwd = 2); points(p_bins, obs_decayed, pch = 16, col = "#ff7f0e")
  legend("topleft", c("recent-only","decayed-full"), col = c("#1f77b4","#ff7f0e"), lwd = 2, bty = "n")
  draw_note()
}

# 4) Copula impact: scatter vs diagonal
plot_copula_scatter <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  set.seed(7)
  n <- 250
  u <- pnorm(scale(rnorm(n)))
  v <- 0.6*u + 0.4*pnorm(scale(rnorm(n)))
  plot(u, v, pch = 16, cex = 0.6, col = rgb(0,0,0,0.35), xlab = "U (margin)", ylab = "V (total)", main = "Copula Pricing Impact (qualitative)")
  abline(0,1, col = "#888888", lty = 2)
  draw_note()
}

# 5) Integer-margin calibration: bars + two lines
plot_integer_margin <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  d <- -12:12
  base <- dnorm(d, mean = 0, sd = 6); base <- base/sum(base)
  rw <- base; keys <- c(3,7,10)
  rw[match(keys, d)] <- rw[match(keys, d)] * 1.8; rw <- rw/sum(rw)
  obs <- rw * 0.9 + base * 0.1
  plot(d, base, type = "l", lwd = 2, col = "#1f77b4", xlab = "Margin d", ylab = "Probability", main = "Integer-Margin Calibration")
  lines(d, rw, lwd = 2, col = "#ff7f0e")
  points(d, obs, pch = 15, cex = 0.8, col = "#000000")
  abline(v = c(3,6,7,10), col = "#cccccc", lty = 3)
  legend("topright", c("baseline","reweighted","observed (mock)"), col = c("#1f77b4","#ff7f0e","#000000"), lwd = c(2,2,NA), pch = c(NA,NA,15), bty = "n")
  draw_note()
}

# 6) Acceptance rates (grouped bars)
plot_acceptance_rates <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  seasons <- 2020:2024
  margin <- c(0.80,0.85,0.83,0.88,0.86)
  keys   <- c(0.78,0.82,0.81,0.85,0.84)
  mat <- rbind(margin, keys)
  barplot(mat, beside = TRUE, names.arg = seasons, col = c("#1f77b4","#ff7f0e"), ylim = c(0,1), ylab = "Pass rate", main = "Simulator Acceptance Rates")
  legend("topleft", c("Margins","Key masses"), fill = c("#1f77b4","#ff7f0e"), bty = "n")
  draw_note()
}

# 7) Acceptance vs live performance: boxplots
plot_acceptance_vs_live <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  set.seed(21)
  clv_fail <- rnorm(100, mean = 5, sd = 15)
  clv_pass <- rnorm(100, mean = 20, sd = 10)
  boxplot(list(Fail = clv_fail, Pass = clv_pass), ylab = "CLV (bps)", main = "Acceptance vs Live Performance")
  draw_note()
}

# 8) Alpha sensitivity: two points/line per method
plot_alpha_sensitivity <- function(path) {
  op <- open_dev(path)
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  alpha <- c(0.05, 0.10)
  growth_kelly <- c(1.8, 1.6)
  growth_rl    <- c(2.1, 1.9)
  plot(alpha, growth_kelly, type = "b", lwd = 2, pch = 16, col = "#1f77b4", xlab = expression(alpha), ylab = "Growth (arbitrary units)", main = "Alpha Sensitivity")
  lines(alpha, growth_rl, type = "b", lwd = 2, pch = 16, col = "#ff7f0e")
  legend("topright", c("Kelly-LCB","RL"), col = c("#1f77b4","#ff7f0e"), lwd = 2, pch = 16, bty = "n")
  draw_note()
}

# Dispatcher
mk <- function(path, title = NULL, subtitle = NULL) {
  # Choose specialized plot where available; else draw a simple title card
  if (grepl("time_decay_weights\\.png$", path)) return(invisible(plot_time_decay(path)))
  if (grepl("rolling_oos_logloss\\.png$", path)) return(invisible(plot_rolling_metric(path, ylab = "Log loss", main = "Rolling OOS Log Loss")))
  if (grepl("rolling_oos_ece\\.png$", path)) return(invisible(plot_rolling_metric(path, ylab = "ECE", main = "Rolling OOS ECE")))
  if (grepl("reliability_curves_timeframe\\.png$", path)) return(invisible(plot_reliability(path)))
  if (grepl("teaser_pricing_copula_delta\\.png$", path)) return(invisible(plot_copula_scatter(path)))
  if (grepl("integer_margin_calibration\\.png$", path)) return(invisible(plot_integer_margin(path)))
  if (grepl("sim_acceptance_rates\\.png$", path)) return(invisible(plot_acceptance_rates(path)))
  if (grepl("sim_acceptance_vs_live_perf\\.png$", path)) return(invisible(plot_acceptance_vs_live(path)))
  if (grepl("alpha_sensitivity_panel\\.png$", path)) return(invisible(plot_alpha_sensitivity(path)))
  # Fallback title card
  op <- open_dev(path, mar = c(2,2,2,2))
  if (is.null(op)) return(invisible(FALSE))
  on.exit(close_dev(op), add = TRUE)
  plot.new()
  text(0.5, 0.65, ifelse(is.null(title), basename(path), title), cex = 1.4, font = 2)
  if (!is.null(subtitle) && nzchar(subtitle)) text(0.5, 0.50, subtitle, cex = 1.0)
  rect(0.15, 0.2, 0.85, 0.4, border = "#888888", col = NA, lwd = 1.5)
  text(0.5, 0.30, "Placeholder: replace via Quarto notebooks", cex = 0.9, col = "#444444")
  invisible(TRUE)
}

placeholders <- list(
  list("analysis/dissertation/figures/out/time_decay_weights.png",         "Time-Decay Weights",                "H ∈ {3,4,5}, centered on 2024"),
  list("analysis/dissertation/figures/out/rolling_oos_logloss.png",        "Rolling OOS Log Loss",             "recent-only vs decayed-full"),
  list("analysis/dissertation/figures/out/rolling_oos_ece.png",            "Rolling OOS ECE",                  "recent-only vs decayed-full"),
  list("analysis/dissertation/figures/out/reliability_curves_timeframe.png","Reliability Curves (2024)",        "recent-only vs decayed-full"),
  list("analysis/dissertation/figures/out/teaser_pricing_copula_delta.png", "Copula Pricing Impact",            "Gaussian vs t"),
  list("analysis/dissertation/figures/out/integer_margin_calibration.png", "Integer-Margin Calibration",       "Observed vs predicted"),
  list("analysis/dissertation/figures/out/sim_acceptance_rates.png",       "Simulator Acceptance Rates",       "by season and test"),
  list("analysis/dissertation/figures/out/sim_acceptance_vs_live_perf.png", "Acceptance vs Live Performance",   "CLV/ROI vs pass/fail"),
  list("analysis/dissertation/figures/out/alpha_sensitivity_panel.png",    "Alpha Sensitivity Panel",          "α ∈ {0.05, 0.10}")
)

created <- vapply(placeholders, function(x) do.call(mk, as.list(x)), logical(1))
invisible(created)
