.onAttach = function(libname, pkgname){
	packageStartupMessage(paste(
		"Welcome to fastHurdleNegBinomRegression v", utils::packageVersion("fastHurdleNegBinomRegression"), ".\n", 
		sep = ""
	))
}