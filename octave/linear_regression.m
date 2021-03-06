function theta = linear_regression(X, y)

	current_loss = 10^10;
	learning_rate = .001;

	m = rows(X);
	theta = rand(m, 1);

	continue_gradient = true

	while(continue_gradient)
		#hypothesis of x parameterized by theta	
		hX = (theta' * X)';

		loss = hX .- y;

		squared_loss = loss .^ 2;
		sum_loss = sum(squared_loss, 'native');

		J_theta = (1 / (2 * m)) * sum_loss;

		#current loss.  compare to previous loss to see if loss has decreased.  if not, do not update theta.
		if( J_theta >= current_loss)
			#early termination
			disp('loss is not decreasing.  terminating')
			return;
		endif

		current_loss = J_theta;

		#update theta
		theta = theta - ((learning_rate / m) * X * (hX - y))
	
	endwhile

endfunction