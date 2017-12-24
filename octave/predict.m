function q = predict(x, theta)
	
	x_predict = [1 x];

	#theta is not invertible. must be multiplied in this order

	hypothesis = theta' * x

	return;
endfunction