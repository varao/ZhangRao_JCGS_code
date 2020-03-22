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

model_parameters:
	utility functions

simulation.py: 
	Main experiment for the 3-dimension immigration model. 

exe_example.sh: 
	Start the main experiment by running bash exe_sample.sh

=============================================
How to run the experiment
example:
	run the experiment with default settings:

	bash exe_sample.sh

	experiment settings:
	dim=3, alpha = 2, beta = 2, sample size = 5000

---------------------------------------------
	run the experiment with customized settings:

	bash exe_sample.sh 3 3 2 10000

	experiment settings:
	dim=3, alpha = 3, beta = 2, sample size = 10000
=============================================

	
