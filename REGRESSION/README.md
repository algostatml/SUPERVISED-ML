# Regularized regression

Closed form solution and optimized solution to solving the Ridge and lasso regression.

----------------------
##### Ridge (L2) Regression | [code](https://github.com/algostatml/SUPERVISED-ML/tree/master/REGRESSION/Ridge%20Regression)

##### Ridge regression is best known for its ability to reduce the complexity of a model using the L2-norm (or euclidean norm). This happens by shricking the coefficients of the model towards zero. It turns out Ridge regression works well for dense dataset and learns a bad hypothesis for non-sparse problem.
----------------------
##### Lasso (L1) Regression | [code](https://github.com/algostatml/SUPERVISED-ML/tree/master/REGRESSION/Lasso%20Regression)
##### [Grafting Lasso Algorithm](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/GraftinglassoAlgo.py) | [Shooting Lasso Algorithm](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/shootinglassoAlgo.py) | [Lasso via eta-trick (GD & SGD)](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/eta_tricklasso.py)
##### Lasso regression solves the problem of regularization and induces sparsity for dense feature space. Its requires the L1-norm which is infact a non-convex and non-differentaible norm. Using different algorithms proposed by researchers in recent times, we are able to transform the L1-norm into a convex and differentiable form. The solution to this algorithm is the backbone behind this lasso.
