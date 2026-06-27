#!/usr/bin/env Rscript

# Generate reference test data for jolie distributions using R.
# Requires: jsonlite
#
# Ground truth: R's d/p/q for built-in distributions, closed-form moments
# computed analytically in this script.
# Caveats:
#   - digits = 17 in toJSON over-specifies but guarantees f64 round-trip.

suppressPackageStartupMessages(library(jsonlite))

# Anchor paths off the script location. Must be invoked via `Rscript <path>`,
# not sourced.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", script_args, value = TRUE)
if (length(file_arg) != 1) {
  stop("must be invoked via `Rscript scripts/generate_reference_data.R` (found no --file= arg)")
}
script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg)))
REPO <- dirname(script_dir)
UNIV_ROOT <- file.path(REPO, "src", "distributions", "univariate")
if (!dir.exists(UNIV_ROOT)) {
  stop(sprintf("expected repo layout at %s; got nothing. Check invocation path.", UNIV_ROOT))
}

# ── Helpers ─────────────────────────────────────────────────────────────

safe_num <- function(v) {
  # `any(...)` keeps this scalar-safe even if a vector slips in: bare `||` on a
  # length>1 logical is a hard error on R >= 4.3.
  if (length(v) == 0 || any(is.na(v)) || any(!is.finite(v))) NA_real_ else v
}

make_moments <- function(mean, variance, skewness, kurtosis, entropy, mode = NA_real_) {
  list(
    mean     = unbox(safe_num(mean)),
    variance = unbox(safe_num(variance)),
    skewness = unbox(safe_num(skewness)),
    kurtosis = unbox(safe_num(kurtosis)),
    entropy  = unbox(safe_num(entropy)),
    mode     = unbox(safe_num(mode))
  )
}

# Point evaluations. `ccdf_fn` / `log_cdf_fn` are optional — only continuous
# distributions expose `ccdf`, and a distribution that lacks either method
# simply omits the field rather than emitting a value nothing reads.
make_point_evals <- function(xs, pdf_fn, cdf_fn, log_pdf_fn, ccdf_fn = NULL, log_cdf_fn = NULL) {
  lapply(xs, function(xi) {
    out <- list(
      x       = unbox(xi),
      pdf     = unbox(safe_num(pdf_fn(xi))),
      cdf     = unbox(safe_num(cdf_fn(xi))),
      log_pdf = unbox(safe_num(log_pdf_fn(xi)))
    )
    if (!is.null(ccdf_fn))    out$ccdf    <- unbox(safe_num(ccdf_fn(xi)))
    if (!is.null(log_cdf_fn)) out$log_cdf <- unbox(safe_num(log_cdf_fn(xi)))
    out
  })
}

make_quantiles <- function(probs, quantile_fn) {
  lapply(probs, function(p) {
    list(
      p = unbox(p),
      x = unbox(safe_num(quantile_fn(p)))
    )
  })
}

write_json <- function(data, path) {
  json_str <- toJSON(data, pretty = TRUE, na = "null", digits = 17)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(paste0(json_str, "\n"), path, sep = "")
  cat(sprintf("  Written: %s\n", path))
}

# ── Distributions ───────────────────────────────────────────────────────

generate_uniform <- function(out_root) {
  parameterizations <- list(
    c(0.0, 1.0),
    c(-10.0, 10.0),
    c(0.001, 0.002),
    c(-100.0, -50.0),
    c(0.0, 1000.0)
  )
  quantile_probs <- c(0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

  cases <- lapply(parameterizations, function(ab) {
    a <- ab[1]; b <- ab[2]
    width <- b - a

    moments <- make_moments(
      mean     = (a + b) / 2,
      variance = width^2 / 12,
      skewness = 0.0,
      kurtosis = -6/5,
      entropy  = log(width),
      mode     = (a + b) / 2
    )

    points <- c(a - width, a - 0.001, a, a + width * 0.25,
                a + width * 0.5, a + width * 0.75, b, b + 0.001, b + width)

    pdf_fn     <- function(x) dunif(x, min = a, max = b)
    cdf_fn     <- function(x) punif(x, min = a, max = b)
    log_pdf_fn <- function(x) dunif(x, min = a, max = b, log = TRUE)
    ccdf_fn    <- function(x) punif(x, min = a, max = b, lower.tail = FALSE)
    log_cdf_fn <- function(x) punif(x, min = a, max = b, log.p = TRUE)
    quant_fn   <- function(p) qunif(p, min = a, max = b)

    log_pdf_below <- dunif(a - width, min = a, max = b, log = TRUE)
    log_pdf_above <- dunif(b + width, min = a, max = b, log = TRUE)

    list(
      params     = list(a = unbox(a), b = unbox(b)),
      moments    = moments,
      pdf_cdf    = make_point_evals(points, pdf_fn, cdf_fn, log_pdf_fn, ccdf_fn, log_cdf_fn),
      quantiles  = make_quantiles(quantile_probs, quant_fn),
      edge_cases = list(
        pdf_nan               = unbox(NA_real_),
        cdf_neg_inf           = unbox(punif(-Inf, min = a, max = b)),
        cdf_pos_inf           = unbox(punif(Inf,  min = a, max = b)),
        log_pdf_below_support = unbox(safe_num(log_pdf_below)),
        log_pdf_above_support = unbox(safe_num(log_pdf_above))
      )
    )
  })

  data <- list(distribution = unbox("Uniform"), cases = cases)
  write_json(data, file.path(out_root, "continuous", "uniform", "test_reference.json"))
}

generate_discrete_uniform <- function(out_root) {
  parameterizations <- list(
    c(0L, 9L), c(1L, 6L), c(-5L, 5L), c(0L, 0L), c(0L, 100L), c(-100L, -50L)
  )
  # Skip p=0.0 for discrete: SciPy ppf(0) returns below-support by convention
  quantile_probs <- c(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

  cases <- lapply(parameterizations, function(ab) {
    a <- ab[1]; b <- ab[2]
    n <- b - a + 1

    # Closed-form moments for DiscreteUniform(a, b). Skew/kurtosis undefined
    # when n=1 (variance = 0); mode is convention-dependent → lower bound.
    skew_val <- if (n == 1) NA_real_ else 0.0
    kurt_val <- if (n == 1) NA_real_ else -6 * (n^2 + 1) / (5 * (n^2 - 1))
    moments <- make_moments(
      mean     = (a + b) / 2,
      variance = (n^2 - 1) / 12,
      skewness = skew_val,
      kurtosis = kurt_val,
      entropy  = log(n),
      mode     = a
    )

    support_points <- if (n <= 20) {
      a:b
    } else {
      sort(unique(c(a, a + 1, a + n %/% 4, a + n %/% 2, a + 3 * n %/% 4, b - 1, b)))
    }
    points <- as.numeric(c(a - 2L, a - 1L, support_points, b + 1L, b + 2L))

    in_support <- function(x) x >= a & x <= b & x == floor(x)
    pdf_fn     <- function(x) ifelse(in_support(x), 1 / n, 0)
    cdf_fn     <- function(x) {
      if (x < a) 0 else if (x >= b) 1 else (floor(x) - a + 1) / n
    }
    log_pdf_fn <- function(x) ifelse(in_support(x), -log(n), -Inf)
    ccdf_fn    <- function(x) 1 - cdf_fn(x)
    # log_cdf = log(cdf); cdf == 0 below support → -Inf → null (skipped).
    log_cdf_fn <- function(x) log(cdf_fn(x))
    # Quantile: smallest k such that P(X <= k) >= p. The driver compares this to
    # Rust's inverse_cdf with `assert_eq!` (exact integer), so the two ceilings
    # must agree. This uses no fp tolerance while Rust's inverse_cdf subtracts a
    # small one; they match for all probs/n here, but a new `quantile_probs`
    # value where `p * n` lands just above an integer could split them.
    quant_fn   <- function(p) {
      if (p <= 0) a else if (p >= 1) b else a + ceiling(p * n) - 1
    }

    list(
      params     = list(a = unbox(as.numeric(a)), b = unbox(as.numeric(b))),
      moments    = moments,
      pdf_cdf    = make_point_evals(points, pdf_fn, cdf_fn, log_pdf_fn, ccdf_fn, log_cdf_fn),
      quantiles  = make_quantiles(quantile_probs, quant_fn),
      edge_cases = list(
        pdf_nan               = unbox(NA_real_),
        cdf_neg_inf           = unbox(0),
        cdf_pos_inf           = unbox(1),
        log_pdf_below_support = unbox(safe_num(log_pdf_fn(a - 1))),
        log_pdf_above_support = unbox(safe_num(log_pdf_fn(b + 1)))
      )
    )
  })

  data <- list(distribution = unbox("DiscreteUniform"), cases = cases)
  write_json(data, file.path(out_root, "discrete", "discrete_uniform", "test_reference.json"))
}

generate_exponential <- function(out_root) {
  scales <- c(1.0, 0.5, 2.0, 0.01, 100.0, 10.0)
  quantile_probs <- c(0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

  cases <- lapply(scales, function(scale) {
    rate <- 1.0 / scale

    # Analytic closed-form moments for Exp(scale = θ); mode is always 0.
    mean_val <- scale
    std      <- scale
    moments <- make_moments(
      mean     = scale,
      variance = scale^2,
      skewness = 2.0,
      kurtosis = 6.0,               # excess
      entropy  = 1.0 + log(scale),
      mode     = 0.0
    )

    points <- sort(unique(c(
      0.0,
      max(0.01, mean_val * 0.1),
      mean_val * 0.5,
      mean_val,
      mean_val + std,
      mean_val + 2 * std,
      mean_val + 3 * std
    )))

    pdf_fn     <- function(x) dexp(x, rate = rate)
    cdf_fn     <- function(x) pexp(x, rate = rate)
    log_pdf_fn <- function(x) dexp(x, rate = rate, log = TRUE)
    ccdf_fn    <- function(x) pexp(x, rate = rate, lower.tail = FALSE)
    log_cdf_fn <- function(x) pexp(x, rate = rate, log.p = TRUE)
    quant_fn   <- function(p) qexp(p, rate = rate)

    log_pdf_below <- dexp(qexp(0, rate = rate) - 1, rate = rate, log = TRUE)
    log_pdf_above <- dexp(qexp(1, rate = rate) + 1, rate = rate, log = TRUE)

    list(
      params     = list(a = unbox(scale), b = unbox(0.0)),
      moments    = moments,
      pdf_cdf    = make_point_evals(points, pdf_fn, cdf_fn, log_pdf_fn, ccdf_fn, log_cdf_fn),
      quantiles  = make_quantiles(quantile_probs, quant_fn),
      edge_cases = list(
        pdf_nan               = unbox(NA_real_),
        cdf_neg_inf           = unbox(pexp(-Inf, rate = rate)),
        cdf_pos_inf           = unbox(pexp(Inf,  rate = rate)),
        log_pdf_below_support = unbox(safe_num(log_pdf_below)),
        log_pdf_above_support = unbox(safe_num(log_pdf_above))
      )
    )
  })

  data <- list(distribution = unbox("Exponential"), cases = cases)
  write_json(data, file.path(out_root, "continuous", "exponential", "test_reference.json"))
}

generate_normal <- function(out_root) {
  parameterizations <- list(
    c(0.0, 1.0), c(5.0, 2.0), c(0.0, 0.01), c(0.0, 100.0),
    c(-10.0, 3.0), c(1000.0, 50.0)
  )
  quantile_probs <- c(0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

  cases <- lapply(parameterizations, function(ms) {
    mu <- ms[1]; sigma <- ms[2]
    moments <- make_moments(
      mean     = mu,
      variance = sigma^2,
      skewness = 0.0,
      kurtosis = 0.0,                # excess
      entropy  = 0.5 * log(2 * pi * exp(1) * sigma^2),
      mode     = mu
    )

    points <- sort(unique(c(
      mu - 4 * sigma, mu - 3 * sigma, mu - 2 * sigma, mu - sigma, mu - 0.5 * sigma,
      mu, mu + 0.5 * sigma, mu + sigma, mu + 2 * sigma, mu + 3 * sigma, mu + 4 * sigma
    )))

    pdf_fn     <- function(x) dnorm(x, mean = mu, sd = sigma)
    cdf_fn     <- function(x) pnorm(x, mean = mu, sd = sigma)
    log_pdf_fn <- function(x) dnorm(x, mean = mu, sd = sigma, log = TRUE)
    quant_fn   <- function(p) qnorm(p, mean = mu, sd = sigma)

    list(
      params     = list(a = unbox(mu), b = unbox(sigma)),
      moments    = moments,
      pdf_cdf    = make_point_evals(points, pdf_fn, cdf_fn, log_pdf_fn),
      quantiles  = make_quantiles(quantile_probs, quant_fn),
      edge_cases = list(
        pdf_nan               = unbox(NA_real_),
        cdf_neg_inf           = unbox(pnorm(-Inf, mean = mu, sd = sigma)),
        cdf_pos_inf           = unbox(pnorm(Inf,  mean = mu, sd = sigma)),
        log_pdf_below_support = unbox(safe_num(dnorm(qnorm(0, mean = mu, sd = sigma) - 1,
                                                     mean = mu, sd = sigma, log = TRUE))),
        log_pdf_above_support = unbox(safe_num(dnorm(qnorm(1, mean = mu, sd = sigma) + 1,
                                                     mean = mu, sd = sigma, log = TRUE)))
      )
    )
  })

  data <- list(distribution = unbox("Normal"), cases = cases)
  write_json(data, file.path(out_root, "continuous", "normal", "test_reference.json"))
}

generate_lognormal <- function(out_root) {
  parameterizations <- list(
    c(0.0, 1.0), c(1.0, 0.5), c(0.0, 2.0), c(3.0, 1.0), c(-1.0, 0.5), c(5.0, 0.1)
  )
  quantile_probs <- c(0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

  cases <- lapply(parameterizations, function(ms) {
    mu <- ms[1]; sigma <- ms[2]
    s2 <- sigma^2
    mean_val <- exp(mu + s2 / 2)
    variance <- (exp(s2) - 1) * exp(2 * mu + s2)
    std      <- sqrt(variance)
    moments <- make_moments(
      mean     = mean_val,
      variance = variance,
      skewness = (exp(s2) + 2) * sqrt(exp(s2) - 1),
      kurtosis = exp(4 * s2) + 2 * exp(3 * s2) + 3 * exp(2 * s2) - 6,   # excess
      entropy  = mu + log(sigma) + 0.5 * log(2 * pi) + 0.5,
      mode     = exp(mu - s2)
    )

    median_val <- exp(mu)
    points <- sort(unique(c(
      max(0.01, median_val * 0.01),
      max(0.01, median_val * 0.1),
      median_val * 0.5,
      median_val,
      median_val * 2.0,
      median_val * 5.0,
      mean_val + std,
      mean_val + 2 * std
    )))

    pdf_fn     <- function(x) dlnorm(x, meanlog = mu, sdlog = sigma)
    cdf_fn     <- function(x) plnorm(x, meanlog = mu, sdlog = sigma)
    log_pdf_fn <- function(x) dlnorm(x, meanlog = mu, sdlog = sigma, log = TRUE)
    quant_fn   <- function(p) qlnorm(p, meanlog = mu, sdlog = sigma)

    log_pdf_below <- dlnorm(qlnorm(0, meanlog = mu, sdlog = sigma) - 1,
                            meanlog = mu, sdlog = sigma, log = TRUE)
    log_pdf_above <- dlnorm(qlnorm(1, meanlog = mu, sdlog = sigma) + 1,
                            meanlog = mu, sdlog = sigma, log = TRUE)

    list(
      params     = list(a = unbox(mu), b = unbox(sigma)),
      moments    = moments,
      pdf_cdf    = make_point_evals(points, pdf_fn, cdf_fn, log_pdf_fn),
      quantiles  = make_quantiles(quantile_probs, quant_fn),
      edge_cases = list(
        pdf_nan               = unbox(NA_real_),
        cdf_neg_inf           = unbox(plnorm(-Inf, meanlog = mu, sdlog = sigma)),
        cdf_pos_inf           = unbox(plnorm(Inf,  meanlog = mu, sdlog = sigma)),
        log_pdf_below_support = unbox(safe_num(log_pdf_below)),
        log_pdf_above_support = unbox(safe_num(log_pdf_above))
      )
    )
  })

  data <- list(distribution = unbox("LogNormal"), cases = cases)
  write_json(data, file.path(out_root, "continuous", "lognormal", "test_reference.json"))
}

# ── Main ────────────────────────────────────────────────────────────────

main <- function() {
  cat("Generating reference data (R)...\n")
  generate_uniform(UNIV_ROOT)
  generate_exponential(UNIV_ROOT)
  generate_normal(UNIV_ROOT)
  generate_lognormal(UNIV_ROOT)
  generate_discrete_uniform(UNIV_ROOT)
  cat("Done.\n")
}

main()
