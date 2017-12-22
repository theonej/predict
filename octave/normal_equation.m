function theta = normal_equation(X, y)

	#add theta 0
	X = [ones(rows(X), 1) X];

	theta = (pinv(X'*X))*X'*y;

	return;
endfunction
