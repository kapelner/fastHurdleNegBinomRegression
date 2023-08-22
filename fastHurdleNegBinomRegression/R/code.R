assert_numeric_matrix = function(Xmm){
	checkmate::assert_matrix(Xmm)
	checkmate::assert_numeric(Xmm)
}

#' Fast Hurdle Negative Binomial Fit 
#' 
#' Fits a hurdle negative binomial model
#'
#' @param Xmm   						The model.matrix for X (you need to create this yourself before)
#' @param y	    						The count response vector
#' @param drop_collinear_variables   	Should we drop perfectly collinear variables? Default is \code{FALSE} to inform the user of the problem.
#' @param lm_fit_tol					When \code{drop_collinear_variables = TRUE}, this is the tolerance to detect collinearity among predictors.
#' 										We use the default value from \code{base::lm.fit}'s which is 1e-7. If you fit the logistic regression and
#' 										still get p-values near 1 indicating high collinearity, we recommend making this value smaller.
#' @param eps_f   						From RcppNumerical. Iteration stops if \eqn{|f-f'|/|f|<\epsilon_f}{|f-f'|/|f|<eps_f},
#'              						where \eqn{f} and \eqn{f'} are the current and previous value
#'              						of the objective function (negative log likelihood) respectively.
#' @param eps_g							From RcppNumerical. Iteration stops if
#'              						\eqn{||g|| < \epsilon_g * \max(1, ||\beta||)}{||g|| < eps_g * max(1, ||beta||)},
#'           						   	where \eqn{\beta}{beta} is the current coefficient vector and
#'              						\eqn{g} is the gradient.
#' @param maxit							From RcppNumerical. Maximum number of L-BFGS iterations.
#' @param num_cores						Number of cores to use to speed up matrix multiplication and matrix inversion (used only during inference computation). Default is 1.
#' 										Unless the number of variables, i.e. \code{ncol(Xmm)}, is large, there does not seem to be a performance gain in using multiple cores.

#'
#' @return      A list of raw results
#' @useDynLib 							fastHurdleNegBinomRegression, .registration=TRUE
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	 Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
fast_hnb_regression = function(Xmm, y, drop_collinear_variables = FALSE, lm_fit_tol = 1e-7, eps_f = 1e-6, eps_g = 1e-5, maxit = 300L, num_cores = 1){
	assert_numeric_matrix(Xmm)
	assert_vector(y)
	for (y_i in y){
		assert_count(y_i)
	}	
	assert_logical(drop_collinear_variables)
	assert_numeric(lm_fit_tol,  lower = .Machine$double.eps)
	assert_numeric(eps_f,  		lower = .Machine$double.eps)
	assert_numeric(eps_g, 		lower = .Machine$double.eps)
	assert_integer(maxit)
	assert_count(maxit, positive = TRUE)
	assert_count(num_cores, positive = TRUE)
	original_col_names = colnames(Xmm)
	
	p = ncol(Xmm) #the original p before variables are potentially dropped
	
	if (length(y) != nrow(Xmm)){
		stop("The number of rows in Xmm must be equal to the length of ybin")
	}
	
	#create the augmented data and do some convenient splits
	z = as.numeric(y == 0)
	Xmm_y_pos  = Xmm[z == 0, ]
	y_pos      = y[z == 0]
	
	variables_retained = rep(TRUE, p)
	names(variables_retained) = original_col_names
	if (drop_collinear_variables){
		collinear_variables = c()
		#we do fits for both the zeroes and positives
		repeat {
			b = coef(lm.fit(Xmm, z, tol = lm_fit_tol))
			b_NA = b[is.na(b)]
			if (length(b_NA) == 0){
				break
			}
			bad_var = gsub("Xmm", "", names(b_NA)[1])
			Xmm = Xmm[, colnames(Xmm) != bad_var] #remove these bad variable(s) from the data!!
			collinear_variables = c(collinear_variables, bad_var)
		}
		repeat {
			b = coef(lm.fit(Xmm_y_pos, y_pos, tol = lm_fit_tol))
			b_NA = b[is.na(b)]
			if (length(b_NA) == 0){
				break
			}
			bad_var = gsub("Xmm", "", names(b_NA)[1])
			Xmm = Xmm[, colnames(Xmm) != bad_var] #remove these bad variable(s) from the data!!
			collinear_variables = c(collinear_variables, bad_var)
		}
		variables_retained[collinear_variables] = FALSE
	}
	
	#now we get the initial starting vals
	gammas_0 = fastLogisticRegressionWrap::fast_logistic_regression(Xmm, z)$coefficients
	betas_0 = coef(lm.fit(Xmm_y_pos, log(y_pos))) #the neg binomial model fits in log space
	#to get the best guess of the phi we use MM via the formula from https://mc-stan.org/docs/2_20/functions-reference/nbalt.html
	mu_y_hat = mean(y_pos - 1)
	ln_initial_phi = log(mu_y_hat^2 / (var(y_pos - 1) - mu_y_hat))
	
	flr = fast_hnb_cpp(Xmm, y, z, c(gammas_0, betas_0, ln_initial_phi), eps_f, eps_g, maxit) 
	flr$Xmm = Xmm
	flr$y = y
	flr$z = z
	flr$gammas_0 = gammas_0
	flr$betas_0 = betas_0
	flr$ln_initial_phi = ln_initial_phi
	flr$variables_retained = variables_retained
	if (drop_collinear_variables){
		flr$collinear_variables = collinear_variables
		coefs = flr$coefficients #save originals
		flr$coefficients = array(NA, 2 * p + 1)
		flr$coefficients[variables_retained] = coefs #all dropped variables will be NA's
	}
	names(flr$coefficients) = original_col_names
	flr$original_regressor_names = original_col_names
	flr$rank = ncol(Xmm) + ncol(Xmm) + 1 #the gammas, betas and the phi
	flr$deviance = -2 * flr$loglikelihood 
	flr$aic = flr$deviance + 2 * flr$rank

	#return
	class(flr) = "fast_hnb"
	flr
}
