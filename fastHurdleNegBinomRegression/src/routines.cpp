#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppNumerical.h>

using namespace Rcpp;
using namespace Numer;
using namespace std;

typedef Eigen::Map<Eigen::MatrixXd> MapMat;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

class HurdleNegativeBinomialRegression: public MFuncGrad {
private:
    const MapMat X;
    const MapVec y;
    const MapVec z;
    const int n;
    const int p;
    const double ln_two;
public:
    HurdleNegativeBinomialRegression(const MapMat X_, const MapVec y_, const MapVec z_) :
        X(X_),
        y(y_),
        z(z_),
        n(X_.rows()),
		p(X_.cols()),
		ln_two(log(2.0))
    {}

    double ln_one_plus_exp_x(double x, bool use_approx){
    	return ln_two + x / 2 + pow(x, 2) / 8 - pow(x, 4) / 192;
    }

//    double ln_one_plus_exp_minus_x(double x){
//    	return ln_two - x / 2 + pow(x, 2) / 8 - pow(x, 4) / 192;
//    }

    double ln_one_plus_c_times_exp_x(double x, double natlog_c, bool use_approx){
    	return ln_one_plus_exp_x(natlog_c + x);
    }

//    double ln_one_plus_c_times_exp_minus_x(double x, double natlog_c){
//    	return ln_one_plus_exp_x(natlog_c - x);
//    }

    double ln_c_plus_exp_x(double x, double natlog_c, bool use_approx){
    	return natlog_c + ln_one_plus_exp_x(x - natlog_c);
    }

    double inverse_one_plus_exp_x(double x, bool use_approx){
    	return 0.5 - x / 4 + pow(x, 3) / 48 - pow(x, 5) / 480;
    }

//    double inverse_one_plus_exp_minus_x(double x){
//    	return 0.5 + x / 4 - pow(x, 3) / 48 + pow(x, 5) / 480;
//    }

    double inverse_one_plus_c_times_exp_x(double x, double natlog_c, bool use_approx){
    	return inverse_one_plus_exp_x(natlog_c + x);
    }

//    double inverse_one_plus_c_times_exp_minus_x(double x, double natlog_c){
//    	return inverse_one_plus_exp_x(natlog_c - x);
//    }

    double inverse_c_plus_exp_x(double x, double c, bool use_approx){
    	double one_over_c = 1 / c;
    	return one_over_c * inverse_one_plus_c_times_exp_x(x, log(one_over_c));
    }

    //see example here: https://github.com/yixuan/RcppNumerical
    double f_grad(Constvec& thetas, Refvec grad){

    	//pull out beta vec and gamma vec and phi from the running theta estimate vector
    	Eigen::VectorXd gammas(p);
    	for (int j = 0; j < p; j++){
    		gammas[j] = thetas[j];
    	}
    	Eigen::VectorXd betas(p);
    	for (int j = p; j < 2 * p; j++){
    		betas[j - p] = thetas[j];
    	}
    	double phi = thetas[2 * p];

    	Rcout << "  gammas " << endl << gammas << endl << " betas " << endl << betas << endl << " phi " << endl << phi << endl;

    	//first calculate some useful values to cache
    	Eigen::VectorXd etas(n);
//    	Eigen::VectorXd exp_etas(n);
//    	Eigen::VectorXd exp_neg_etas(n);
    	for (int i = 0; i < n; i++){
    		etas[i]     =     X.row(i) * gammas;
//    		exp_etas[i] =     exp(etas[i]);
//    		exp_neg_etas[i] = exp(-etas[i]);
    	}
    	Eigen::VectorXd xis(n);
//    	Eigen::VectorXd exp_xis(n);
//    	Eigen::VectorXd exp_neg_xis(n);
    	for (int i = 0; i < n; i++){
    		xis[i]     =     X.row(i) * betas;
//        	exp_xis[i] =     exp(xis[i]);
//        	exp_neg_xis[i] = exp(-xis[i]);
    	}
    	Rcout << "  etas "     << endl << etas                                                 << endl;
//    	Rcout << "  exp_etas " << endl << exp_etas << " exp_neg_etas " << endl << exp_neg_etas << endl;
    	Rcout << "  xis "      << endl << xis                                                  << endl;
//    	Rcout << "  exp_xis "  << endl << exp_xis  << " exp_neg_xis "  << endl << exp_neg_xis  << endl;

    	//first add up the log likelihood
    	double ln_phi = log(phi);
		double lgamma_phi_minus_two = lgamma(phi - 2);
		Rcout << " lgamma_phi_minus_two " << lgamma_phi_minus_two << endl;
    	double loglik = 0;
    	for (int i = 0; i < n; i++){
    		double y_i = y[i];
    		double eta_i = etas[i];
//    		double eta_i_sq = pow(eta_i, 2);
    		double xi_i = xis[i];
    		Rcout << "  loglik calc i " << i << " y_i " << y_i << " z_i " << z[i];
    		if (z[i] == 1){
    			loglik += -ln_one_plus_exp_x(-eta_i);
    			Rcout << " ln_one_plus_exp_x(-eta_i) " << ln_one_plus_exp_x(-eta_i);
    		} else {
    			loglik += (
    						-ln_one_plus_exp_x(eta_i) +
							lgamma(y_i + phi - 3) - lgamma(y_i) - lgamma_phi_minus_two +
							-(y_i - 1) * ln_one_plus_c_times_exp_x(-xi_i, ln_phi) +
							-phi * ln_one_plus_c_times_exp_x(xi_i, -ln_phi)
						  );
            	Rcout << " ln_one_plus_exp_x(eta_i) " << ln_one_plus_exp_x(eta_i)
            		  << " ln_one_plus_c_times_exp_x(-xi_i, ln_phi) " << ln_one_plus_c_times_exp_x(-xi_i, ln_phi)
					  << " lgamma(y_i) " << lgamma(y_i) << " lgamma(y_i + phi - 3) " << lgamma(y_i + phi - 3)
					  << " ln_one_plus_c_times_exp_x(xi_i, -ln_phi) " << ln_one_plus_c_times_exp_x(xi_i, -ln_phi) << endl;
    		}
        	Rcout << " loglik " << loglik << endl;
    	}

    	//then compute all the 2 * p + 1 partial derivatives (i.e., the entries in the gradient)
    	for (int j = 0; j < (2 * p + 1); j++){ //initialize
    		grad[j] = 0;
    	}
    	Rcpp::Function digamma("digamma");
    	double digamma_phi_minus_two = *REAL(digamma(phi - 2));
    	for (int i = 0; i < n; i++){
    		Eigen::VectorXd x_i = X.row(i);

    		double y_i_minus_one = y[i] - 1;
    		double eta_i = etas[i];
//    		double eta_i_sq = pow(eta_i, 2);
    		double xi_i = xis[i];
//    		double exp_etas_i = exp_etas[i];
//    		double exp_neg_etas_i = exp_neg_etas[i];
//    		double exp_xis_i = exp_xis[i];
//    		double exp_neg_xis_i = exp_neg_xis[i];
        	Rcout << "  grad calc i " << i;

    		if (z[i] == 1){
    			Rcout << endl;
            	for (int j = 0; j < p; j++){
            		grad[j] += x_i(j) * inverse_one_plus_exp_x(eta_i);
            	}
     		} else {
            	Rcout << "  y_i_minus_one " << y_i_minus_one << endl;
            	for (int j = 0; j < p; j++){
            		grad[j] -= x_i(j) * inverse_one_plus_exp_x(-eta_i);
            		grad[j] -= phi * inverse_one_plus_c_times_exp_x(-xi_i, ln_phi);
            	}
            	for (int j = p; j < 2 * p; j++){
            		grad[j] += x_i(j) * y_i_minus_one * inverse_c_plus_exp_x(xi_i,  ln_phi);
            		grad[j] -= x_i(j) * phi * inverse_one_plus_c_times_exp_x(-xi_i, ln_phi);
            	}
            	//now handle phi
            	grad[2 * p] += *REAL(digamma(y[i] + phi - 3)) +
            			-digamma_phi_minus_two +
            			-y_i_minus_one * inverse_c_plus_exp_x(xi_i, ln_phi) +
						 inverse_one_plus_c_times_exp_x(-xi_i, ln_phi) +
						-ln_c_plus_exp_x(xi_i, ln_phi) +
						ln_phi;
     		}
    	}

    	Rcout << "  grad " << endl << grad << endl;
        return -loglik; //we must return the cost as the *negative* log likelihood as the algorithm *minimizes* cost (thus it will maximize likelihood)
    }
};

// inspired by https://github.com/yixuan/RcppNumerical/blob/master/src/fastLR.cpp

// [[Rcpp::export]]
Rcpp::List fast_hnb_cpp(
		Rcpp::NumericMatrix X,
		Rcpp::NumericVector y,
		Rcpp::NumericVector z,
		Rcpp::NumericVector theta_start,
		double eps_f,
		double eps_g,
		int maxit
	){

    const MapMat XX = Rcpp::as<MapMat>(X);
    const MapVec yy = Rcpp::as<MapVec>(y);
    const MapVec zz = Rcpp::as<MapVec>(z);

	// Initial guess
	Rcpp::NumericVector theta_hats = Rcpp::clone(theta_start);
	MapVec thetas(theta_hats.begin(), theta_hats.length());

    // Negative log likelihood
    HurdleNegativeBinomialRegression nll(XX, yy, zz);

	double negloglikelihood;
	int status = optim_lbfgs(nll, thetas, negloglikelihood, maxit, eps_f, eps_g);
	if (status < 0)
		Rcpp::warning("algorithm did not converge");

//	Rcpp::List return_list =
//	if (do_inference){
//
//	}
	return Rcpp::List::create(
			Rcpp::Named("coefficients")      = theta_hats,
			Rcpp::Named("loglikelihood")     = -negloglikelihood,
			Rcpp::Named("converged")         = (status >= 0)
		);
}
