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
public:
    HurdleNegativeBinomialRegression(const MapMat X_, const MapVec y_, const MapVec z_) :
        X(X_),
        y(y_),
        z(y_),
        n(X.rows()),
		p(X.cols())
    {}

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

    	Rcout << "  gammas " << gammas << " betas " << betas << " phi " << phi << endl;

    	//first calculate some useful values to cache
    	Eigen::VectorXd exp_etas(n);
    	Eigen::VectorXd exp_neg_etas(n);
    	for (int i = 0; i < n; i++){
    		double eta_i = X.row(i) * gammas;
    		exp_etas[i] =     exp(eta_i);
    		exp_neg_etas[i] = exp(-eta_i);
    	}
    	Eigen::VectorXd exp_xis(n);
    	Eigen::VectorXd exp_neg_xis(n);
    	for (int i = 0; i < n; i++){
    		double xi_i = X.row(i) * betas;
        	exp_xis[i] =     exp(xi_i);
        	exp_neg_xis[i] = exp(-xi_i);
    	}

    	//first add up the log likelihood
    	double loglik = 0;
    	for (int i = 0; i < n; i++){
    		double inverse_phi = 1 / phi;
    		double lgamma_phi_minus_two = lgamma(phi - 2);
    		double y_i = y[i];
    		if (z[i] == 1){
    			loglik += -log(1 + exp_neg_etas[i]);
    		} else {
    			loglik += -log(1 + exp_etas[i]) +
    					   lgamma(y_i + phi - 3) - lgamma(y_i) - lgamma_phi_minus_two +
						  -(y_i - 1) * log(1 + phi * exp_neg_xis[i]) +
						  -phi * log(1 + inverse_phi * exp_xis[i]);
    		}
    	}

    	//then compute all the 2*p + 1 partial derivatives (i.e., the entries in the gradient)
    	for (int j = 0; j < (2 * p + 1); j++){ //initialize
    		grad[j] = 0;
    	}
    	for (int i = 0; i < n; i++){
        	for (int j = 0; j < p; j++){
        		grad[j] += 1;
        	}
        	for (int j = p; j < 2 * p; j++){
        		grad[j - p] += 1;
        	}
        	grad[2 * p + 1] += 1;
    	}

        return -loglik; //we must return the cost as the *negative* log likelihood as the algorithm *minimizes* cost (thus it will maximize likelihood)
    }
};

// inspired by https://github.com/yixuan/RcppNumerical/blob/master/src/fastLR.cpp

// [[Rcpp::export]]
Rcpp::List fast_hnb_cpp(
		Rcpp::NumericMatrix X,
		Rcpp::NumericVector y,
		Rcpp::IntegerVector z,
        Rcpp::NumericVector theta_start,
        double eps_f,
		double eps_g,
		int maxit){

//    const MapMat XX = ;
//    const MapVec yy = ;
//    const MapVec zz = ;

	// Initial guess
	Rcpp::NumericVector theta_hats = Rcpp::clone(theta_start);
	MapVec thetas(theta_hats.begin(), theta_hats.length());

    // Negative log likelihood
    HurdleNegativeBinomialRegression nll(Rcpp::as<MapMat>(X), Rcpp::as<MapVec>(y), Rcpp::as<MapVec>(z));

	double negloglikelihood;
	int status = optim_lbfgs(nll, thetas, negloglikelihood, maxit, eps_f, eps_g);
	if (status < 0)
		Rcpp::warning("algorithm did not converge");

	return Rcpp::List::create(
		Rcpp::Named("coefficients")      = theta_hats,
		Rcpp::Named("loglikelihood")     = -negloglikelihood,
		Rcpp::Named("converged")         = (status >= 0)
	);
}
