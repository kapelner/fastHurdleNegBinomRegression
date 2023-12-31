assert_binary_vector_then_cast_to_numeric = function(vec){
  checkmate::assert_choice(class(vec), c("numeric", "integer", "logical"))
  vec = as.numeric(vec)
  if (!(checkmate::testSetEqual(unique(vec), c(0, 1)) | checkmate::testSetEqual(unique(vec), c(0)) | checkmate::testSetEqual(unique(vec), c(1)))){ #binary only
	  stop("Set must consist of zeroes and/or ones.")
  }
  vec
}

assert_numeric_matrix = function(Xmm){
  checkmate::assert_matrix(Xmm)
  checkmate::assert_numeric(Xmm)
}

#' FastLR Wrapper
#' 
#' Returns most of what you get from glm
#'
#' @param Xmm   						The model.matrix for X (you need to create this yourself before)
#' @param ybin  						The binary response vector
#' @param drop_collinear_variables   	Should we drop perfectly collinear variables? Default is \code{FALSE} to inform the user of the problem.
#' @param lm_fit_tol					When \code{drop_collinear_variables = TRUE}, this is the tolerance to detect collinearity among predictors.
#' 										We use the default value from \code{base::lm.fit}'s which is 1e-7. If you fit the logistic regression and
#' 										still get p-values near 1 indicating high collinearity, we recommend making this value smaller.
#' @param do_inference_on_var			Which variables should we compute approximate standard errors of the coefficients and approximate p-values for the test of
#' 										no linear log-odds probability effect? Default is \code{"none"} for inference on none (for speed). If not default, then \code{"all"}
#' 										to indicate inference should be computed for all variables. The final option is to pass one index to indicate the column
#' 										number of \code{Xmm} where inference is desired. We have a special routine to compute inference for one variable only. It consists of a conjugate
#' 										gradient descent which is another approximation atop the coefficient-fitting approximation in RcppNumerical. Note: if you are just comparing
#' 										nested models using anova, there is no need to compute inference for coefficients (keep the default of \code{FALSE} for speed).
#' @param Xt_times_diag_w_times_X_fun	A custom function whose arguments are \code{X} (an n x m matrix), \code{w} (a vector of length m) and this function's \code{num_cores} 
#' 										argument in that order. The function must return an m x m R matrix class object which is the result of the computing X^T %*% diag(w) %*% X. If your custom  
#' 										function is not parallelized, the \code{num_cores} argument is ignored. Default is \code{NULL} which uses the function 
#' 										\code{\link{eigen_Xt_times_diag_w_times_X}} which is implemented with the Eigen C++ package and hence very fast. The only way we know of to beat the default is to use a method that employs
#' 										GPUs. See README on github for more information.
#' @param sqrt_diag_matrix_inverse_fun	A custom function that returns a numeric vector which is square root of the diagonal of the inverse of the inputted matrix. Its arguments are \code{X} 
#' 										(an n x n matrix) and this function's \code{num_cores} argument in that order. If your custom function is not parallelized, the \code{num_cores} argument is ignored. 
#' 										The object returned must further have a defined function \code{diag} which returns the diagonal of the matrix as a vector. Default is \code{NULL} which uses the function 
#' 										\code{\link{eigen_inv}} which is implemented with the Eigen C++ package and hence very fast. The only way we know of to beat the default is to use a method that employs
#' 										GPUs. See README on github for more information.
#' @param num_cores						Number of cores to use to speed up matrix multiplication and matrix inversion (used only during inference computation). Default is 1.
#' 										Unless the number of variables, i.e. \code{ncol(Xmm)}, is large, there does not seem to be a performance gain in using multiple cores.
#' @param ...   						Other arguments to be passed to \code{fastLR}. See documentation there.
#'
#' @return      A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	 Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
fast_logistic_regression = function(Xmm, ybin, drop_collinear_variables = FALSE, lm_fit_tol = 1e-7, do_inference_on_var = "none", Xt_times_diag_w_times_X_fun = NULL, sqrt_diag_matrix_inverse_fun = NULL, num_cores = 1, ...){
  assert_numeric_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  assert_logical(drop_collinear_variables)
  assert_numeric(lm_fit_tol, lower = 0)
  assert_function(Xt_times_diag_w_times_X_fun, null.ok = TRUE, args = c("X", "w", "num_cores"), ordered = TRUE, nargs = 3)
  assert_function(sqrt_diag_matrix_inverse_fun, null.ok = TRUE, args = c("X", "num_cores"), ordered = TRUE, nargs = 2)
  assert_count(num_cores, positive = TRUE)
  original_col_names = colnames(Xmm)
  
  p = ncol(Xmm) #the original p before variables are dropped
  
  assert_choice(class(do_inference_on_var), c("character", "numeric", "integer"))
  if (is(do_inference_on_var, "character")){
	  assert_choice(do_inference_on_var, c("none", "all"))
	  do_inference_on_var_name = NULL
  } else {
	  assert_count(do_inference_on_var, positive = TRUE)
	  assert_numeric(do_inference_on_var, upper = p)
	  do_inference_on_var_name = original_col_names[do_inference_on_var]
  }
  do_any_inference = do_inference_on_var != "none"
  
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  #cat("ncol Xmm:", ncol(Xmm), "\n")
  #cat("rank Xmm:", Matrix::rankMatrix(Xmm), "\n")
	
  variables_retained = rep(TRUE, p)
  names(variables_retained) = original_col_names
  if (drop_collinear_variables){
	  collinear_variables = c()
	  repeat {
		  b = coef(lm.fit(Xmm, ybin, tol = lm_fit_tol))
		  b_NA = b[is.na(b)]
		  if (length(b_NA) == 0){
			  break
		  }
		  bad_var = gsub("Xmm", "", names(b_NA)[1])
		  #cat("bad_var", bad_var, "\n")
		  Xmm = Xmm[, colnames(Xmm) != bad_var] #remove these bad variable(s) from the data!!
		  collinear_variables = c(collinear_variables, bad_var)
	  }
	  #if (length(collinear_variables) > 1){
	  #	  warning(paste("Dropped the following variables due to collinearity:\n", paste0(collinear_variables, collapse = ", ")))
	  #}	  
	  #cat("ncol Xmm after:", ncol(Xmm), "\n")
	  #cat("rank Xmm after:", Matrix::rankMatrix(Xmm), "\n")
	  #b = coef(lm.fit(Xmm, ybin, tol = lm_fit_tol))
	  #print(b)
	  #solve(t(Xmm) %*% Xmm, tol = inversion_tol)
	  if (do_any_inference & !is.null(do_inference_on_var_name)){
		  if (do_inference_on_var_name %in% collinear_variables){
			  warning("There is no longer any inference to compute as the variables specified was collinear and thus dropped from the model fit.")
			  do_any_inference = FALSE			  
		  }
	  }
	  variables_retained[collinear_variables] = FALSE
  }
  
  flr = RcppNumerical::fastLR(Xmm, ybin, ...)
  flr$Xmm = Xmm
  flr$ybin = ybin
  flr$do_inference_on_var = do_inference_on_var
  flr$variables_retained = variables_retained
  if (drop_collinear_variables){
	flr$collinear_variables = collinear_variables
	coefs = flr$coefficients #save originals
	flr$coefficients = array(NA, p)
	flr$coefficients[variables_retained] = coefs #all dropped variables will be NA's
  }
  names(flr$coefficients) = original_col_names
  flr$original_regressor_names = original_col_names
  flr$rank = ncol(Xmm)
  flr$deviance = -2 * flr$loglikelihood 
  flr$aic = flr$deviance + 2 * flr$rank
  

  if (do_any_inference){
	  b = flr$coefficients[variables_retained]  
	  
	  flr$se = 						array(NA, p)
	  flr$z = 						array(NA, p)
	  flr$approx_pval = 			array(NA, p)
	  names(flr$se) =   			original_col_names
	  names(flr$z) =   				original_col_names
	  names(flr$approx_pval) =   	original_col_names
	  
	  #compute the std errors of the coefficient estimators 
	  #we compute them via notes found in https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture26.pdf
	  exp_Xmm_dot_b = exp(Xmm %*% b)
	  w = as.numeric(exp_Xmm_dot_b / (1 + exp_Xmm_dot_b)^2)
	  XmmtWmatXmm =   if (is.null(Xt_times_diag_w_times_X_fun)){
						  eigen_Xt_times_diag_w_times_X(Xmm, w, num_cores) #t(Xmm) %*% diag(w) %*% Xmm
					  } else {
						  Xt_times_diag_w_times_X_fun(Xmm, w, num_cores) #t(Xmm) %*% diag(w) %*% Xmm
					  }
	  
	  
	  if (do_inference_on_var == "all"){
		  tryCatch({ #compute the entire inverse (this could probably be sped up by only computing the diagonal a la https://web.stanford.edu/~lexing/diagonal.pdf but I have not found that implemented anywhere)
			  sqrt_diag_XmmtWmatXmminv =  if (is.null(sqrt_diag_matrix_inverse_fun)){
											sqrt(diag(eigen_inv(XmmtWmatXmm, num_cores)))
										  } else {
										  	sqrt_diag_matrix_inverse_fun(XmmtWmatXmm, num_cores)
										  }
		  }, 
		  error = function(e){
			  print(e)
			  stop("Error in inverting X^T X.\nTry setting drop_collinear_variables = TRUE\nto automatically drop perfectly collinear variables.\n")
		  })
		  
		  flr$se[variables_retained] = sqrt_diag_XmmtWmatXmminv
	  } else { #only compute the one entry of the inverse that is of interest. Right now this is too slow to be useful but eventually it will be implemente via:
		flr$se[do_inference_on_var] = sqrt(eigen_compute_single_entry_of_diagonal_matrix(XmmtWmatXmm, do_inference_on_var, num_cores))
	  }

	  flr$z[variables_retained] = 				b / flr$se[variables_retained]
	  flr$approx_pval[variables_retained] = 	2 * pnorm(-abs(flr$z[variables_retained]))
  }

  #return
  class(flr) = "fast_logistic_regression"
  flr
}

#' FastLR Wrapper Summary
#' 
#' Returns the summary table a la glm
#'
#' @param object       The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param alpha_order  Should the coefficients be ordered in alphabetical order? Default is \code{TRUE}.
#' @param ...          Other arguments to be passed to \code{summary}.
#'
#' @return             The summary as a data.frame
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' summary(flr)
summary.fast_logistic_regression = function(object, alpha_order = TRUE, ...){
  checkmate::assert_choice(class(object), c("fast_logistic_regression", "fast_logistic_regression_stepwise"))
  checkmate::assert_logical(alpha_order)
  if (!object$converged){
      warning("fast LR did not converge")
  }
  if (object$do_inference_on_var == "none"){
	  cat("please refit the model with the \"do_inference_on_var\" argument set to \"all\" or a single variable index number.\n")
  } else {
	  df = data.frame(
	    approx_coef = object$coefficients,
	    approx_se = object$se,
	    approx_z = object$z,
	    approx_pval = object$approx_pval,
	    signif = ifelse(is.na(object$approx_pval), "", ifelse(object$approx_pval < 0.001, "***", ifelse(object$approx_pval < 0.01, "**", ifelse(object$approx_pval < 0.05, "*", ""))))
	  )
	  rownames(df) = object$original_regressor_names
	  if (alpha_order){
		  df = df[order(rownames(df)), ]
	  }
	  df
  }
}

#' FastLR Wrapper Summary
#' 
#' Returns the summary table a la glm
#'
#' @param object     The object built using the \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...        Other arguments to be passed to \code{summary}.
#'
#' @return           The summary as a data.frame
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' summary(flr)
summary.fast_logistic_regression_stepwise = function(object, ...){
	checkmate::assert_class(object, "fast_logistic_regression_stepwise")
	summary(object$flr, ...)
}

#' FastLR Wrapper Print
#' 
#' Returns the summary table a la glm
#'
#' @param x     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...   Other arguments to be passed to print
#' 
#' @return      The summary as a data.frame
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' print(flr)
print.fast_logistic_regression = function(x, ...){
	summary(x, ...)
}

#' FastLR Wrapper Print
#' 
#' Returns the summary table a la glm
#'
#' @param x     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param ...   Other arguments to be passed to print
#' 
#' @return      The summary as a data.frame
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#' 	Xmm = model.matrix(~ . - type, Pima.te), 
#'  ybin = as.numeric(Pima.te$type == "Yes"))
#' print(flr)
print.fast_logistic_regression_stepwise = function(x, ...){
	summary(x$flr, ...)
}


#' FastLR Wrapper Predictions
#' 
#' Predicts returning p-hats
#'
#' @param object     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param newdata    A matrix of observations where you wish to predict the binary response.
#' @param type       The type of prediction required. The default is \code{"response"} which is on the response scale (i.e. probability estimates) and the alternative is \code{"link"} which is the linear scale (i.e. log-odds).
#' @param ...        Further arguments passed to or from other methods
#' 
#' @return           A numeric vector of length \code{nrow(newdata)} of estimates of P(Y = 1) for each unit in \code{newdata}.
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
predict.fast_logistic_regression = function(object, newdata, type = "response", ...){
  checkmate::assert_class(object, "fast_logistic_regression")
  assert_numeric_matrix(newdata)
  checkmate::assert_choice(type, c("link", "response"))
  
  #if new_data has more features than training data, we can subset it
  old_data_features = object$original_regressor_names
  newdata = newdata[, old_data_features]
  
  #now we need to make sure newdata is legal
  new_data_features = colnames(newdata)
  if (length(new_data_features) != length(old_data_features)){
    stop("newdata has ", length(new_data_features), " features and training data has ", length(old_data_features))
  }
  # new_features_minus_old_features = setdiff(new_data_features, old_data_features)
  # if (length(setdiff(new_features_minus_old_features)) > 0){
  #   stop("newdata must have same columns as the original training data matrix in the same order.\nHere, newdata has features\n", paste(new_features_minus_old_features, collapse = ", "), "\nwhich training data did not have")
  # }
  new_features_minus_old_features = setdiff(old_data_features, new_data_features)
  if (!all(colnames(newdata) == old_data_features)){
    stop("newdata must have same columns as the original training data matrix in the same order.\nHere, training data has features\n", paste(new_features_minus_old_features, collapse = ", "), "\nwhich newdata did not have")
  }
  if (!object$converged){
    warning("fast LR did not converge")
  }
  b = object$coefficients
  b[is.na(b)] = 0 #this is the way to ignore NA's
  log_odds_predictions = c(newdata %*% b)
  if (type == "response"){
    exp_Xmm_dot_b = exp(log_odds_predictions)
    exp_Xmm_dot_b / (1  + exp_Xmm_dot_b)
  } else if (type == "link"){
    log_odds_predictions
  }
}

#' FastLR Wrapper Predictions
#' 
#' Predicts returning p-hats
#'
#' @param object     The object built using the \code{fast_logistic_regression} or \code{fast_logistic_regression_stepwise} wrapper functions
#' @param newdata    A matrix of observations where you wish to predict the binary response.
#' @param type       The type of prediction required. The default is \code{"response"} which is on the response scale (i.e. probability estimates) and the alternative is \code{"link"} which is the linear scale (i.e. log-odds).
#' @param ...        Further arguments passed to or from other methods
#' 
#' @return           A numeric vector of length \code{nrow(newdata)} of estimates of P(Y = 1) for each unit in \code{newdata}.
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
predict.fast_logistic_regression_stepwise = function(object, newdata, type = "response", ...){	
	checkmate::assert_class(object, "fast_logistic_regression_stepwise")
	predict.fast_logistic_regression(object$flr, newdata, type = "response", ...)
}

#' Rapid Forward Stepwise Logistic Regression
#' 
#' Roughly duplicates the following \code{glm}-style code:
#' 
#'  \code{nullmod = glm(ybin ~ 0,     data.frame(Xmm), family = binomial)}
#'  \code{fullmod = glm(ybin ~ 0 + ., data.frame(Xmm), family = binomial)}
#'  \code{forwards = step(nullmod, scope = list(lower = formula(nullmod), upper = formula(fullmod)), direction = "forward", trace = 0)}
#'
#' @param Xmm             			The model.matrix for X (you need to create this yourself before).
#' @param ybin            			The binary response vector.
#' @param mode						"aic" (default, fast) or "pval" (slow, but possibly yields a better model).
#' @param pval_threshold  			The significance threshold to include a new variable. Default is \code{0.05}.
#' 									If \code{mode == "aic"}, this argument is ignored.
#' @param use_intercept   			Should we automatically begin with an intercept? Default is \code{TRUE}.
#' @param drop_collinear_variables 	Parameter used in \code{fast_logistic_regression}. Default is \code{FALSE}. See documentation there.
#' @param lm_fit_tol	  			Parameter used in \code{fast_logistic_regression}. Default is \code{1e-7}. See documentation there.
#' @param verbose         			Print out messages during the loop? Default is \code{TRUE}.
#' @param ...             			Other arguments to be passed to \code{fastLR}. See documentation there.
#'
#' @return                			A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' flr = fast_logistic_regression_stepwise_forward(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = as.numeric(Pima.te$type == "Yes")
#' )
fast_logistic_regression_stepwise_forward = function(
		Xmm, 
		ybin, 
		mode = "aic",
		pval_threshold = 0.05, 
		use_intercept = TRUE, 
		verbose = TRUE, 
		drop_collinear_variables = FALSE, 
		lm_fit_tol = 1e-7, 
		...){
  assert_numeric_matrix(Xmm)
  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
  if (length(ybin) != nrow(Xmm)){
    stop("The number of rows in Xmm must be equal to the length of ybin")
  }
  assert_choice(mode, c("aic", "pval"))
  mode_is_aic = (mode == "aic")
  if (!mode_is_aic){
	  assert_numeric(pval_threshold, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
  }
  
  assert_logical(use_intercept)
  assert_logical(verbose)
  
  #create starting point
  n = nrow(Xmm)
  p = ncol(Xmm)
  if (use_intercept){
    if (unique(Xmm[, 1]) == 1){
      Xmmt = Xmm[, 1, drop = FALSE]
      js = 1
      iter = 1
      if (verbose){
        cat("iteration #", iter, "of possibly", p, "added intercept", "\n")
      }
    } else {
      Xmmt = matrix(1, nrow = n, ncol = 1)
      colnames(Xmmt) = "(Intercept)"
      js = 0
      iter = 0
    }
  } else {
    Xmmt = matrix(NA, nrow = n, ncol = 0)
    js = 0
    iter = 0
  }
  if (mode_is_aic){
	  aics_star = c()
	  last_aic_star = .Machine$double.xmax #anything will beat this
  } else {
	  pvals_star = c()
  }
  
  repeat {
    js_to_try = setdiff(1 : p, js)
    if (length(js_to_try) == 0){
      break
    }
	if (mode_is_aic){
		aics = array(NA, p)
	} else {
		pvals = array(NA, p)	
	}
    
    for (i_j in 1 : length(js_to_try)){
      j = js_to_try[i_j]
      Xmmtemp = Xmmt
      Xmmtemp = cbind(Xmmtemp, Xmm[, j, drop = FALSE])
      # tryCatch({
		ptemp = ncol(Xmmtemp)
		do_inference_on_var = 	if (mode_is_aic){
									"none"
								} else {
									"all" #I don't think single variable is ready for primetime yet
								}
        flrtemp = fast_logistic_regression(Xmmtemp, ybin, drop_collinear_variables, lm_fit_tol, do_inference_on_var = do_inference_on_var)
		if (mode_is_aic){
			aics[j] = flrtemp$aic
		} else {
			if (!is.null(flrtemp$approx_pval)){ #if the last variable got dropped due to collinearity, we skip this
				pvals[j] = flrtemp$approx_pval[ptemp] #the last one
			}			
		}
	
		if (verbose){
			cat("   sub iteration #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in ")
			if (mode_is_aic){
				cat("aic", aics[j], "\n")
			} else {
				cat("pval", pvals[j], "\n")	
			}
			
		}
      # }, error = function(e){
      #   cat("   iter #", i_j, "of", length(js_to_try), "with feature", colnames(Xmm)[j], "resulted in ERROR\n")
      # })
    }
	if (mode_is_aic){
		if (min(aics, na.rm = TRUE) > last_aic_star){
			break
		}
	} else {
		if (!any(pvals < pval_threshold, na.rm = TRUE)){
			break
		}
	}

	j_star = 	if (mode_is_aic){
					which.min(aics)
				} else {
					which.min(pvals)
				}
	
	#if (is.na(j_star) | is.null(j_star) | is.na(aics[j_star])){
	#	stop("j_star problem")
	#}
    js = c(js, j_star)
	if (mode_is_aic){
		aics_star = c(aics_star, aics[j_star])
		last_aic_star = aics[j_star]
	} else {
		pvals_star = c(pvals_star, pvals[j_star])
	}	
    
    Xmmt = cbind(Xmmt, Xmm[, j_star, drop = FALSE])
    
    iter = iter + 1
    if (verbose){
      cat("iteration #", iter, "of possibly", p, "added feature #", j_star, "named", colnames(Xmm)[j_star], "with ")
	  if (mode_is_aic){
		  cat("aic", aics[j_star], "\n")
	  } else {
		  cat("pval", pvals[j_star], "\n")
	  }					  
    }
  }
  #return some information you would like to see
  flr_stepwise = list(js = js, flr = fast_logistic_regression(Xmmt, ybin, drop_collinear_variables, lm_fit_tol, do_inference_on_var = "all"))
  if (mode_is_aic){
	  flr_stepwise$aics = aics
  } else {
	  flr_stepwise$pvals_star = pvals_star
  }  
  class(flr_stepwise) = "fast_logistic_regression_stepwise"
  flr_stepwise
}

#' Binary Confusion Table and Errors
#' 
#' Provides a binary confusion table and error metrics
#'
#' @param yhat            		The binary predictions
#' @param ybin            		The true binary responses
#' @param skip_argument_checks	If \code{TRUE} it does not check this function's arguments for appropriateness. It is not recommended unless you truly need speed and thus the default is \code{FALSE}.
#'
#' @return                		A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' ybin = as.numeric(Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
#' confusion_results(phat > 0.5, ybin)
confusion_results = function(yhat, ybin, skip_argument_checks = FALSE){  
  n = length(yhat)
  if (!skip_argument_checks){
	  assert_logical(skip_argument_checks)
	  yhat = assert_binary_vector_then_cast_to_numeric(yhat)
	  ybin = assert_binary_vector_then_cast_to_numeric(ybin)
	  if (n != length(ybin)){
		  stop("yhat and ybin must be same length")
	  }	  
  }

  conf = fast_two_by_two_binary_table_cpp(ybin, yhat)
  tp = conf[2, 2]
  tn = conf[1, 1]
  fp = conf[1, 2]
  fn = conf[2, 1]
  
  fdr =  fp / sum(conf[, 2])
  fomr = fn / sum(conf[, 1])
  fpr = fp / sum(conf[1, ])
  fnr = fn / sum(conf[2, ])
  
  confusion_sums = matrix(NA, 3, 3)
  confusion_sums[1 : 2, 1 : 2] = conf
  confusion_sums[1, 3] = tn + fp
  confusion_sums[2, 3] = fn + tp
  confusion_sums[3, 3] = n
  confusion_sums[3, 1] = tn + fn
  confusion_sums[3, 2] = fp + tp
  colnames(confusion_sums) = c("0", "1", "sum")
  rownames(confusion_sums) = c("0", "1", "sum")
  
  confusion_proportion_and_errors = matrix(NA, 4, 4)
  confusion_proportion_and_errors[1 : 3, 1 : 3] = confusion_sums / n
  confusion_proportion_and_errors[1, 4] = fpr
  confusion_proportion_and_errors[2, 4] = fnr
  confusion_proportion_and_errors[4, 1] = fomr
  confusion_proportion_and_errors[4, 2] = fdr
  confusion_proportion_and_errors[4, 4] = (fp + fn) / n  
  colnames(confusion_proportion_and_errors) = c("0", "1", "proportion", "error_rate")
  rownames(confusion_proportion_and_errors) = c("0", "1", "proportion", "error_rate")
  
  list(
    confusion_sums = confusion_sums,
    confusion_proportion_and_errors = confusion_proportion_and_errors
  )
}

#' Asymmetric Cost Explorer
#' 
#' Given a set of desired proportions of predicted outcomes, what is the error rate for each of those models?
#' 
#' @param phat                  The vector of probability estimates to be thresholded to make a binary decision
#' @param ybin            		The true binary responses
#' @param steps					All possibile thresholds which must be a vector of numbers in (0, 1). Default is \code{seq(from = 0.001, to = 0.999, by = 0.001)}.
#' @param outcome_of_analysis   Which class do you care about performance? Either 0 or 1 for the negative class or positive class. Default is \code{0}.
#' @param proportions_desired 	Which proportions of \code{outcome_of_analysis} class do you wish to understand performance for? 
#' @param proportion_tolerance  If the model cannot match the proportion_desired within this amount, it does not return that model's performance. Default is \code{0.01}.
#' @param K_folds				If not \code{NULL}, this indicates that we wish to fit the \code{phat} thresholds out of sample using this number of folds. Default is \code{NULL} for in-sample fitting.
#' @return 						A table with column 1: \code{proportions_desired}, column 2: actual proportions (as close as possible), column 3: error rate, column 4: probability threshold.
#' 
#' @author Adam Kapelner
#' @export
asymmetric_cost_explorer = function(
	phat, 
	ybin,
	steps = seq(from = 0.001, to = 0.999, by = 0.001), 
	outcome_of_analysis = 0, 
	proportions_desired = seq(from = 0.1, to = 0.9, by = 0.1),
	proportion_tolerance = 0.01
){
	checkmate::assert_vector(phat)
	checkmate::assert_numeric(phat)
	ybin = assert_binary_vector_then_cast_to_numeric(ybin)
	checkmate::assert(length(phat) == length(ybin))
	checkmate::assert_numeric(steps, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
	checkmate::assert_choice(outcome_of_analysis, c(0, 1))
	checkmate::assert_numeric(proportions_desired, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
	checkmate::assert_numeric(proportion_tolerance, lower = .Machine$double.eps, upper = 1 - .Machine$double.eps)
	num_steps = length(steps)
	temp_res = matrix(NA, nrow = num_steps, ncol = 4)
	
	for (i in 1 : num_steps){
		phat_threshold = steps[i]
		temp_res[i, 1] = phat_threshold 		
		tryCatch({
			conf_tab = confusion_results(phat > phat_threshold, ybin, skip_argument_checks = TRUE)$confusion_proportion_and_errors
			temp_res[i, 2] = conf_tab[3, outcome_of_analysis + 1]
			temp_res[i, 3] = conf_tab[4, outcome_of_analysis + 1]	
			temp_res[i, 4] = conf_tab[4, 4]				
		}, error = function(e){
			reached_error = TRUE
		})
}	
	
##	half_num_steps = round(num_steps / 2)
##     reached_error = FALSE
##     for (i in (half_num_steps : 1)){
##         phat_threshold = steps[i]
##         temp_res[i, 1] = phat_threshold 
## #		if (!reached_error){			
##             tryCatch({
##                 conf_tab = confusion_results(phat > phat_threshold, ybin, skip_argument_checks = TRUE)$confusion_proportion_and_errors
##                 temp_res[i, 2] = conf_tab[3, outcome_of_analysis + 1]
##                 temp_res[i, 3] = conf_tab[4, outcome_of_analysis + 1]	
##                 temp_res[i, 4] = conf_tab[4, 4]				
##             }, error = function(e){
##                 reached_error = TRUE
##             })
## #		}
##     }
## 
##     reached_error = FALSE
##     for (i in ((half_num_steps + 1) : num_steps)){
##         phat_threshold = steps[i]
##         temp_res[i, 1] = phat_threshold 
## #		if (!reached_error){			
##             tryCatch({
##                 conf_tab = confusion_results(ybin, phat > phat_threshold, skip_argument_checks = TRUE)$confusion_proportion_and_errors
##                 temp_res[i, 2] = conf_tab[3, outcome_of_analysis + 1]
##                 temp_res[i, 3] = conf_tab[4, outcome_of_analysis + 1]	
##                 temp_res[i, 4] = conf_tab[4, 4]
##             }, error = function(e){
##                 reached_error = TRUE
##             })
## #		}
##     }	
	
	res = data.frame(
		proportions_desired = proportions_desired,
		actual_proportion = NA,
		fomr = NA,
		miscl_err = NA,
		phat_threshold = NA
	)
	for (k in 1 : length(proportions_desired)){
		abs_diffs = abs(proportions_desired[k] - temp_res[, 2])
		idx = which.min(abs_diffs)
		if (abs_diffs[idx] < proportion_tolerance){
			res[k, 2 : 5] = temp_res[idx, c(2, 3, 4, 1)]
		}
	}
	res
}

#' Asymmetric Cost Explorer
#' 
#' Given a set of desired proportions of predicted outcomes, what is the error rate for each of those models?
#' 
#' @param phat                  The vector of probability estimates to be thresholded to make a binary decision
#' @param ybin            		The true binary responses
#' @param K_CV					We wish to fit the \code{phat} thresholds out of sample using this number of folds. Default is \code{5}.
#' @param ...					Other parameters to be passed into the \code{asymmetric_cost_explorer} function
#' @return 						A table with column 1: \code{proportions_desired}, column 2: actual proportions (as close as possible), column 3: error rate, column 4: probability threshold.
#' 
#' @author Adam Kapelner
#' @export
asymmetric_cost_explorer_cross_validated = function(phat, ybin,	K_CV = 5, ...){
	checkmate::assert_vector(phat)
	checkmate::assert_numeric(phat)
	ybin = assert_binary_vector_then_cast_to_numeric(ybin)
	checkmate::assert(length(phat) == length(ybin))
	checkmate::assert_count(K_CV, positive = TRUE)
	
	res = NULL
	oos_conf_tables_by_proportion = list()
	proportions_desired = NULL
	
	n = length(phat)
	temp = rnorm(n)
	k_fold_idx = cut(temp, breaks = quantile(temp, seq(0, 1, length.out = K_CV + 1)), include.lowest = TRUE, labels = FALSE)
	
	for (k_cv in 1 : K_CV){
		test_idx = which(k_fold_idx == k_cv)
		train_idx = setdiff(1 : n, test_idx)
#		in_sample_res_k = asymmetric_cost_explorer(phat[train_idx], ybin[train_idx])
		in_sample_res_k = asymmetric_cost_explorer(phat[train_idx], ybin[train_idx], ...)
		phat_threshold_k = in_sample_res_k$phat_threshold
		if (is.null(res)){
			proportions_desired = in_sample_res_k$proportions_desired
			#we can't initialize the critical data structures until we know how many rows it needs
			res = data.frame(
				proportions_desired = proportions_desired,
				actual_proportion = NA,
				fomr = NA,
				miscl_err = NA
			)
			for (prop_desired in proportions_desired){
				oos_conf_tables_by_proportion[[as.character(prop_desired)]] = NA 
			}
		}
		
		for (i_thres in 1 : length(phat_threshold_k)){
			phat_i_thres = phat_threshold_k[i_thres]
			prop_desired_i_thres = proportions_desired[i_thres]
			if (!is.na(phat_i_thres)){
				if (is(oos_conf_tables_by_proportion[[as.character(prop_desired_i_thres)]], "logical")){
					oos_conf_tables_by_proportion[[as.character(prop_desired_i_thres)]] = matrix(0, 2, 2)
				}
				oos_conf_tables_by_proportion[[as.character(prop_desired_i_thres)]] = oos_conf_tables_by_proportion[[as.character(prop_desired_i_thres)]] +
					confusion_results(phat[test_idx] > phat_i_thres, ybin[test_idx], skip_argument_checks = TRUE)$confusion_sums[1 : 2, 1 : 2]
			}			
		}
	}
		
	#now tabulate the final results
	for (i_prop in 1 : length(proportions_desired)){
		conf_tab = oos_conf_tables_by_proportion[[as.character(proportions_desired[i_prop])]]
		if (!is(conf_tab, "logical")){
			res[i_prop, "actual_proportion"] = sum(conf_tab[, 1]) / sum(conf_tab)
			res[i_prop, "fomr"] =              sum(conf_tab[2, 1]) / sum(conf_tab[, 1])
			res[i_prop, "miscl_err"] =         (conf_tab[1, 2] + conf_tab[2, 1])/ sum(conf_tab)
		}
	}
	res
}

#' General Confusion Table and Errors
#' 
#' Provides a confusion table and error metrics for general factor vectors.
#' There is no need for the same levels in the two vectors.
#'
#' @param yhat            				The factor predictions
#' @param yfac            				The true factor responses
#' @param proportions_scaled_by_column	When returning the proportion table, scale by column? Default is \code{FALSE} to keep the probabilities 
#' 										unconditional to provide the same values as the function \code{confusion_results}. Set to \code{TRUE}
#' 										to understand error probabilities by prediction bucket.
#'
#' @return                				A list of raw results
#' @export
#' @examples
#' library(MASS); data(Pima.te)
#' ybin = as.numeric(Pima.te$type == "Yes")
#' flr = fast_logistic_regression(
#'   Xmm = model.matrix(~ . - type, Pima.te), 
#'   ybin = ybin
#' )
#' phat = predict(flr, model.matrix(~ . - type, Pima.te))
#' yhat = array(NA, length(ybin))
#' yhat[phat <= 1/3] = "no"
#' yhat[phat >= 2/3] = "yes"
#' yhat[is.na(yhat)] = "maybe"
#' general_confusion_results(factor(yhat, levels = c("no", "yes", "maybe")), factor(ybin)) 
#' #you want the "no" to align with 0, the "yes" to align with 1 and the "maybe" to be 
#' #last to align with nothing
general_confusion_results = function(yhat, yfac, proportions_scaled_by_column = FALSE){
	assert_factor(yhat)
	assert_factor(yfac)
	n = length(yhat)
	if (n != length(yfac)){
		stop("yhat and yfac must be same length")
	}
	levels_yfac = levels(yfac)
	levels_yhat = levels(yhat)
	n_r_conf = length(levels_yfac)
	n_c_conf = length(levels_yhat)
	conf = matrix(table(yfac, yhat), ncol = n_c_conf, nrow = n_r_conf)
	rownames(conf) = levels_yfac
	colnames(conf) = levels_yhat
	
	confusion_sums = matrix(NA, n_r_conf + 1, n_c_conf + 1)
	confusion_sums[1 : n_r_conf, 1 : n_c_conf] = conf
	confusion_sums[n_r_conf + 1, 1 : n_c_conf] = colSums(conf)
	confusion_sums[1 : n_r_conf, n_c_conf + 1] = rowSums(conf)
	confusion_sums[n_r_conf + 1, n_c_conf + 1] = n
	rownames(confusion_sums) = c(levels_yfac, "sum")
	colnames(confusion_sums) = c(levels_yhat, "sum")
	
	confusion_proportion_and_errors = matrix(NA, n_r_conf + 2, n_c_conf + 2)
	if (proportions_scaled_by_column){
		p = ncol(conf)
		for (j in 1 : p){
			confusion_proportion_and_errors[1 : n_r_conf, j] = conf[, j] / sum(conf[, j])		
		}
		confusion_proportion_and_errors[1 : (n_r_conf + 1), p + 1] = confusion_sums[, p + 1] / sum(confusion_sums[, p + 1])
		confusion_proportion_and_errors[(n_r_conf + 1), 1 : (n_c_conf + 1)] = confusion_sums[(n_r_conf + 1), ] / n
	} else {
		confusion_proportion_and_errors[1 : (n_r_conf + 1), 1 : (n_c_conf + 1)] = confusion_sums / n
	}
	
	
	#now calculate all types of errors
	p = min(dim(conf))
	n_correct_classifications = 0
	for (j in 1 : p){
		n_correct_classifications = n_correct_classifications + conf[j, j]
	}	
	for (j in 1 : n_r_conf){
		if (j <= p){
			j_row_sum = sum(conf[j, ])
			confusion_proportion_and_errors[j, n_c_conf + 2] = (j_row_sum - conf[j, j]) / j_row_sum
		} else {
			confusion_proportion_and_errors[j, n_c_conf + 2] = 1
		}
		
	}	
	for (j in 1 : n_c_conf){
		if (j <= p){
			j_col_sum = sum(conf[, j])
			confusion_proportion_and_errors[n_r_conf + 2, j] = (j_col_sum - conf[j, j]) / j_col_sum
		} else {
			confusion_proportion_and_errors[n_r_conf + 2, j] = 1
		}		
	}
	confusion_proportion_and_errors[n_r_conf + 2, n_c_conf + 2] = (n - n_correct_classifications) / n 
	rownames(confusion_proportion_and_errors) = c(levels_yfac, "proportion", "error_rate")
	colnames(confusion_proportion_and_errors) = c(levels_yhat, "proportion", "error_rate")
	
	list(
		confusion_sums = confusion_sums,
		confusion_proportion_and_errors = confusion_proportion_and_errors
	)
}

#' A fast Xt [times] diag(w) [times] X function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size n x p 
#' @param w 				A numeric vector of length p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The resulting matrix 
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   n = 100
#'   p = 10
#'   X = matrix(rnorm(n * p), nrow = n, ncol = p)
#'   w = rnorm(p)
#'   eigen_Xt_times_diag_w_times_X(t(X), w)
eigen_Xt_times_diag_w_times_X = function(X, w, num_cores = 1){
	assert_numeric_matrix(X)
	assert_numeric(w)
	assert_true(nrow(X) == length(w))
	assert_count(num_cores, positive = TRUE)
	
	eigen_Xt_times_diag_w_times_X_cpp(X, w, num_cores)
}

#' Compute Single Value of the Diagonal of a Symmetric Matrix's Inverse
#' 
#' Via the eigen package's conjugate gradient descent algorithm.
#' 
#' @param M 			The symmetric matrix which to invert (and then extract one element of its diagonal)
#' @param j 			The diagonal entry of \code{M}'s inverse
#' @param num_cores 	The number of cores to use. Default is 1.
#' 
#' @return 				The value of m^{-1}_{j,j}
#' 
#' @author Adam Kapelner
#' @export
#' @examples
#' 	n = 500
#' 	X = matrix(rnorm(n^2), nrow = n, ncol = n)
#' 	M = t(X) %*% X
#' 	j = 137
#' 	eigen_compute_single_entry_of_diagonal_matrix(M, j)
#' 	solve(M)[j, j] #to ensure it's the same value

eigen_compute_single_entry_of_diagonal_matrix = function(M, j, num_cores = 1){
	assert_numeric_matrix(M)
	assert_true(ncol(M) == nrow(M))
	assert_count(j, positive = TRUE)
	assert_numeric(j, upper = nrow(M))
	assert_count(num_cores, positive = TRUE)
	
	eigen_compute_single_entry_of_diagonal_matrix_cpp(M, j, num_cores)
}

#' A fast solve(X) function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size p x p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The resulting matrix 
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   p = 10
#'   eigen_inv(matrix(rnorm(p^2), nrow = p))
eigen_inv = function(X, num_cores = 1){
	assert_numeric_matrix(X)
	assert_true(ncol(X) == nrow(X))
	assert_count(num_cores, positive = TRUE)
	
	eigen_inv_cpp(X, num_cores)
}

#' A fast det(X) function
#' 
#' Via the eigen package
#' 
#' @param X					A numeric matrix of size p x p
#' @param num_cores 		The number of cores to use. Unless p is large, keep to the default of 1.
#' 
#' @return					The determinant as a scalar numeric value
#' 
#' @useDynLib 				fastLogisticRegressionWrap, .registration=TRUE
#' @export
#' @examples
#'   p = 30
#'   eigen_det(matrix(rnorm(p^2), nrow = p))
eigen_det = function(X, num_cores = 1){
	assert_numeric_matrix(X)
	assert_true(ncol(X) == nrow(X))
	assert_count(num_cores, positive = TRUE)
	
	eigen_det_cpp(X, num_cores)
}
