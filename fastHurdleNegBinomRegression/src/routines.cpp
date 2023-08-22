#include <Rcpp.h>
#include <RcppEigen.h>
#include <RcppNumerical.h>
#include <fstream>

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
    ofstream& outfile;

public:
    HurdleNegativeBinomialRegression(const MapMat X_, const MapVec y_, const MapVec z_, ofstream& _outfile) :
        X(X_),
        y(y_),
        z(z_),
        n(X_.rows()),
		p(X_.cols()),
		ln_two(log(2.0)),
		outfile(_outfile)
    {}

    double ln_one_plus_exp_x(double x, bool use_approx){
    	if (use_approx){
        	return ln_two + x / 2 + pow(x, 2) / 8 - pow(x, 4) / 192;
    	}
		return log(1 + exp(x));
    }

    double ln_one_plus_c_times_exp_x(double x, double natlog_c, bool use_approx){
    	if (use_approx){
        	return ln_one_plus_exp_x(natlog_c + x, true);
    	}
		return log(1 + exp(natlog_c + x));
    }

    double ln_c_plus_exp_x(double x, double natlog_c, bool use_approx){
    	if (use_approx){
        	return natlog_c + ln_one_plus_exp_x(x - natlog_c, true);
    	}
		return log(exp(natlog_c) + exp(x));
    }

    double inverse_one_plus_exp_x(double x, bool use_approx){
    	if (use_approx){
        	return 0.5 - x / 4 + pow(x, 3) / 48 - pow(x, 5) / 480;
    	}
		return 1 / (1 + exp(x));
    }

    double inverse_one_plus_c_times_exp_x(double x, double natlog_c, bool use_approx){
    	if (use_approx){
        	return inverse_one_plus_exp_x(natlog_c + x, true);
    	}
		return 1 / (1 + exp(natlog_c + x));
    }

    double inverse_c_plus_exp_x(double x, double c, bool use_approx){
    	if (use_approx){
        	double one_over_c = 1 / c;
        	return one_over_c * inverse_one_plus_c_times_exp_x(x, log(one_over_c), true);
    	}
		return 1 / (c + exp(x));
    }

    //see example here: https://github.com/yixuan/RcppNumerical
    double f_grad(Constvec& thetas, Refvec grad){
    	outfile << endl << endl << endl << "================== NEW L-BFGS ITERATION ====================" << endl;

    	bool use_approx = false;

    	//pull out beta vec and gamma vec and phi from the running theta estimate vector
    	Eigen::VectorXd gammas(p);
    	for (int j = 0; j < p; j++){
    		gammas[j] = thetas[j];
    	}
    	Eigen::VectorXd betas(p);
    	for (int j = p; j < 2 * p; j++){
    		betas[j - p] = thetas[j];
    	}
    	double ln_phi = thetas[2 * p];
    	double phi = exp(ln_phi);
    	outfile << "  gammas " << endl << gammas << endl << " betas " << endl << betas << endl << " ln_phi " << endl << ln_phi << " phi " << phi << endl;

    	//first calculate some useful values to cache
    	Eigen::VectorXd etas(n);
    	for (int i = 0; i < n; i++){
    		etas[i]     =     X.row(i) * gammas;
    	}
    	Eigen::VectorXd xis(n);
    	for (int i = 0; i < n; i++){
    		xis[i]     =     X.row(i) * betas;
    	}

    	//first add up the log likelihood
		double lgamma_phi_minus_two = lgamma(phi - 2);
		outfile << " lgamma_phi_minus_two " << lgamma_phi_minus_two << endl;
    	double loglik = 0;
    	for (int i = 0; i < n; i++){
    		double y_i = y[i];
    		double eta_i = etas[i];
    		double xi_i = xis[i];
    		outfile << "  loglik calc i " << i << " y_i " << y_i << " z_i " << z[i] << " loglik " << loglik << endl;
    		if (z[i] == 1){
    			loglik += -ln_one_plus_exp_x(-eta_i, use_approx);
    			outfile << " ln_one_plus_exp_x(-eta_i) " << ln_one_plus_exp_x(-eta_i, use_approx) << " loglik " << loglik << endl;
    		} else {
    			loglik += (
    						-ln_one_plus_exp_x(eta_i, use_approx) +
							lgamma(y_i + phi - 3) - lgamma(y_i) - lgamma_phi_minus_two +
							-(y_i - 1) * ln_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx) +
							-phi * ln_one_plus_c_times_exp_x(xi_i, -ln_phi, use_approx)
						  );
            	outfile << " ln(1 + exp(eta_i)) " << ln_one_plus_exp_x(eta_i, use_approx)
            		    << " ln(1 + phi * exp(-xi_i)) " << ln_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx)
					    << " lgamma(y_i) " << lgamma(y_i) << " lgamma(y_i + phi - 3) " << lgamma(y_i + phi - 3)
					    << " ln(1 + phi * exp(xi_i) " << ln_one_plus_c_times_exp_x(xi_i, -ln_phi, use_approx)
					    << " loglik " << loglik << endl;
    		}
    	}
    	outfile << " final iteration loglik " << loglik << endl;
		outfile << endl;

    	//then compute all the 2 * p + 1 partial derivatives (i.e., the entries in the gradient)
    	Eigen::VectorXd temp_grad(2 * p + 1);
    	for (int j = 0; j < (2 * p + 1); j++){ //initialize
    		temp_grad[j] = 0;
    	}
    	Rcpp::Function digamma("digamma");
    	double digamma_phi_minus_two = *REAL(digamma(phi - 2));
    	for (int i = 0; i < n; i++){
    		const Eigen::VectorXd x_i = X.row(i);

    		const double y_i_minus_one = y[i] - 1;
    		const double eta_i = etas[i];
    		const double xi_i = xis[i];

    		outfile << "grad calc i " << i << " eta_i " << eta_i << " xi_i " << xi_i << " z_i " << z[i] << " y_i_minus_one " << y_i_minus_one << endl;
    		if (z[i] == 1){
    			outfile << endl;
            	for (int j = 0; j < p; j++){
            		outfile << " i " << i << " j = gamma_" << j
						  << " x_i(j) " << x_i(j)
            		      << " 1/(1 + exp(eta_i)) " << inverse_one_plus_exp_x(eta_i, use_approx)
						  << " grad " << temp_grad[j] << " --> ";
            		temp_grad[j] += x_i(j) * inverse_one_plus_exp_x(eta_i, use_approx);
            		outfile << temp_grad[j] << endl;
            	}
     		} else {
            	outfile << "  y_i_minus_one " << y_i_minus_one << endl;
            	for (int j = 0; j < p; j++){
            		outfile << " i " << i << " j = gamma_" << j
						  << " x_i(j) " << x_i(j)
            			  << " 1/(1 + exp(-eta_i)) " << inverse_one_plus_exp_x(-eta_i, use_approx)
						  << " 1/(1 + phi * exp(-eta_i)) " << inverse_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx)
						  << " grad " << temp_grad[j] << " --> ";
            		temp_grad[j] -= x_i(j) * inverse_one_plus_exp_x(-eta_i, use_approx);
            		outfile << temp_grad[j] << " --> ";
            		temp_grad[j] -= phi * inverse_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx);
            		outfile << temp_grad[j] << endl;
            	}
            	for (int j = p; j < 2 * p; j++){
            		outfile << " i " << i << " j = beta_" << (j - p)
						  << " x_i(j) " << x_i(j - p)
						  << " 1/(c + exp(xi_i)) " << inverse_c_plus_exp_x(xi_i, ln_phi, use_approx)
						  << " 1/(1 + phi * exp(-xi_i)) " << inverse_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx)
            			  << " grad " << temp_grad[j] << " --> ";
            		if (y_i_minus_one > 0){
            			temp_grad[j] -= x_i(j- p) * y_i_minus_one * phi * inverse_c_plus_exp_x(xi_i, ln_phi, use_approx);
                		outfile << temp_grad[j] << " --> ";
            		}
            		temp_grad[j] -= x_i(j - p) * phi * inverse_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx);
            		outfile << temp_grad[j] << endl;
            	}
            	//now handle phi
        		outfile << " i " << i << " j = phi" " grad " << temp_grad[2 * p] << " --> ";
            	temp_grad[2 * p] += *REAL(digamma(y[i] + phi - 3));
        		outfile << temp_grad[2 * p] << " --> ";
            	temp_grad[2 * p] -= digamma_phi_minus_two;
        		outfile << temp_grad[2 * p] << " --> ";
            	if (y_i_minus_one > 0){
            		temp_grad[2 * p] -= y_i_minus_one * inverse_c_plus_exp_x(xi_i, ln_phi, use_approx);
            		outfile << temp_grad[2 * p] << " --> ";
            	}
            	temp_grad[2 * p] += inverse_one_plus_c_times_exp_x(-xi_i, ln_phi, use_approx);
        		outfile << temp_grad[2 * p] << " --> ";
            	temp_grad[2 * p] -= ln_c_plus_exp_x(xi_i, ln_phi, use_approx);
        		outfile << temp_grad[2 * p] << " --> ";
            	temp_grad[2 * p] += ln_phi;
        		outfile << temp_grad[2 * p] << " --mult_phi--> ";
        		temp_grad[2 * p] *= phi;
        		outfile << temp_grad[2 * p] << endl;
     		}
    		outfile << endl;
    	}
    	//flip the sign as it's for negative log likelihood
    	for (int j = 0; j < (2 * p + 1); j++){
    		temp_grad[j] = -temp_grad[j];
    	}

    	grad.noalias() = temp_grad;

    	outfile << "  final iteration grad " << endl << grad << endl;
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

	remove("log_lbfgs.log");
	ofstream outfile;
	outfile.open("log_lbfgs.log");
    // Negative log likelihood
    HurdleNegativeBinomialRegression nll(XX, yy, zz, outfile);

	double negloglikelihood;
	int status = optim_lbfgs(nll, thetas, negloglikelihood, maxit, eps_f, eps_g);
	if (status < 0)
		Rcpp::warning("algorithm did not converge");

	outfile.close();

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
