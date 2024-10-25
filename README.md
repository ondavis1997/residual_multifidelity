# residual_multifidelity

-directories sec5.1, sec5.2, sec5.3, sec5.4 correspond to paper sections

-reproducing results

sec5.1/code

	- run main_RMFNN_resnet.py
	- run main_DiscrepNN_resnet.py

	- run time can very long (up to several days depending on available computational resources)
 
sec5.1/post_processing_and_plots

	-run pp.py

#############################################################################################################################
sec5.2/code

	- run main_RMFNN_resnet.py 
	- run main_MFNN_resnet.py
	- run main_HFNN_resnet.py

	- runtime can be very long (up to days depending on available computational resources)

sec5.2/post_processing_and_plots

	-run pp.py

###############################################################################################################################
sec5.3/code

	- run RMFNN_ODE_parallel.py  # this is an mpi program (run time can be very long; days-weeks depending on computational resources)

sec5.3/post_processing_and_plots

	-run pp.py

	- exact timings for training and evaluation will depend on computing architecture

sec5.4/code

	- run RMFNN_PDE_parallel.py  # this is an mpi program (run time can be very long; days-weeks depending on computational resources)

	- exact timings for training and evaluation will depend on computing architecture

sec5.3/post_processing_and_plots

	-run pp.py



