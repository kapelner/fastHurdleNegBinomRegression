# setwd("C:/Users/kapel/workspace/fastHurdleNegBinomRegression")
# library(roxygen2); roxygenise("fastHurdleNegBinomRegression", clean = TRUE)
# setwd("C:/Users/kapel/workspace/fastHurdleNegBinomRegression/fastHurdleNegBinomRegression"); Rcpp::compileAttributes()
# rstudioapi::restartSession()

pacman::p_load(fastHurdleNegBinomRegression, microbenchmark)

set.seed(1984)

n = 100
p = 3
X = cbind(1, matrix(runif(n * p, -1, 1), nrow = n, ncol = p))
gammas = rep(-1, p + 1)
betas  = rep(0.5, p + 1)
phi = 3

p_is = (1 + exp(-X %*% gammas))^-1
mu_is = exp(X %*% betas)

y_is = array(NA, n)
u_is = runif(n)
#create the hurdle data
for (i in 1 : n){
  y_is[i] = if (u_is[i] < p_is[i]){
              0
            } else {
              rnbinom(1, mu = mu_is[i], size = phi)
            }
}
table(y_is)

#standardize X
# Z = apply(X, 2, function(xj){(xj - mean(xj)) / sd(xj)})[, -1]

hnb_mod = fastHurdleNegBinomRegression::fast_hnb_regression(X, y_is, maxit = 2L)
# hnb_mod$Xmm %*% hnb_mod$gammas_0
# hnb_mod$Xmm %*% hnb_mod$betas_0
