function theta = linear_regression(X, y)

	current_loss = 10^10;
	learning_rate = .001;

	m = rows(X);
	theta = rand(m, 1);

	continue_gradient = true

	while(continue_gradient)
		#hypothesis of x parameterized by theta	
		hx = (theta' * X)';

		loss = hx .- y;

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
		for j = [1:rows(theta)]
			theta_j = theta(j);

			theta_loss = 0;
			for i = [1:m]
				#losses have already been calulated for all hx - y
				loss_i = loss(i);
				x = X(:, i);
				xj = x(j);
				theta_loss += xj * loss_i;
			endfor

			theta_j = theta_j - (learning_rate * theta_loss);
			theta(j) = theta_j;
		endfor
	
	endwhile

endfunction