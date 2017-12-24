function theta = logistic_regression(X, y)

	current_loss = 10^10;
	learning_rate = .01;
	maximum_epochs = 100000;

	m = rows(X);
	theta = rand(m, 1);

	continue_gradient = true

	epoch = 0;
	while(continue_gradient)
		#hypothesis of x parameterized by theta	
		Z = (theta' * X)';

		#the sigmoid function (also known as the logisitic function) will normalize our predictions to 0 < y^ < 1
		hX = arrayfun(@sigmoid, Z);

		J_theta = (1 / m) * ((-y' * log(hX)) - ((1-y)' * log(1 - hX)));

		#current loss.  compare to previous loss to see if loss has decreased.  if not, do not update theta.
		if( J_theta >= current_loss)
			#early termination
			disp('loss is not decreasing.  terminating')
			return;
		endif

		current_loss = J_theta;

		#update theta
		theta = theta - ((learning_rate / m) * X * (hX - y))
		
		epoch+=1;
		if(epoch >= maximum_epochs)
			continue_gradient = false;
		endif
	endwhile

endfunction