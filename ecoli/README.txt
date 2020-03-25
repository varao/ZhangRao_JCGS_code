============================================
required python package: 
	numpy version 1.10.0
=============================================
path_class.py: 
	Markov jump process class and observation class
FFBS.py:
	Forward filtering, backward sampling, virtual jump sampling functions in Rao and The 2013

MH.py: 
	Metropolis Hastings sampling function

GBS.py: 
	Gibbs sampling function

model_parameters:
	utility functions

simulation.py: 
	Main experiment for the ecoli data. 

exe_example.sh: 
	Start the main experiment by running bash exe_sample.sh

=============================================
How to run the experiment
example:
	run the experiment with default settings:

	bash exe_sample.sh

	experiment settings:
	sample size = 2000, sample size for computing covariance matrix = 2000

---------------------------------------------
	run the experiment with customized settings:

	bash exe_sample.sh 5000 2000

	experiment settings:
	sample size = 5000, sample size for computing covariance matrix = 2000
=============================================

	
