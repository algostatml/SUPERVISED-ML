# SUPERVISED-ML
Supervised machine learning algorithms explained with code

----------------------
##### Lasso (L1) Regression | [code](https://github.com/algostatml/SUPERVISED-ML/tree/master/REGRESSION/Lasso%20Regression)
##### [Grafting Lasso Algorithm](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/GraftinglassoAlgo.py) | [Shooting Lasso Algorithm](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/shootinglassoAlgo.py) | [Lasso via eta-trick (GD & SGD)](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Lasso%20Regression/eta_tricklasso.py)
##### Lasso regression solves the problem of regularization and induces sparsity for dense feature space. Its requires the L1-norm which is infact a non-convex and non-differentaible norm. Using different algorithms proposed by researchers in recent times, we are able to transform the L1-norm into a convex and differentiable form. The solution to this algorithm is the backbone behind this lasso.

----------------------
##### Ridge (L2) Regression | [code](https://github.com/algostatml/SUPERVISED-ML/tree/master/REGRESSION/Ridge%20Regression)

##### Ridge regression is best known for its ability to reduce the complexity of a model using the L2-norm (or euclidean norm). This happens by shricking the coefficients of the model towards zero. It turns out Ridge regression works well for dense dataset and learns a bad hypothesis for non-sparse problem.

##### Kernel Ridge Regression | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Ridge%20Regression/KERNEL%20RIDGE/kernelridge.py)

##### Kernel regression is a machine learning preceduce for projecting 2-Dimension sapce into N-Dimensional space. We achieve this by satisfying the Mercer's Theory of inner products. Kernel ridge regression project the low dimensional feature space to higher dimension space by using the concept of support vector (representer theorem).

----------------------
##### Logistic Regression | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/LogisticRegression.py)

##### Here we present the solution to logistic regression using the cross-entropy loss function. This is a binary classifier for supervised classification problem and we solve the optimization function derived from the maximum likelihood using gradient ascent.

----------------------
##### Kernel Logistic Regression | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/KERNEL%20LOGISTIC%20REGRESSION/kernelLogistic.py)

##### Logistic regression is a linear binary classifier as implemented here. We therefore extend this version to handle non-linear datatypes by kernelizing it. The implementation method here uses Gradient Descent and Iterative Reweighted Least Square (IRLS) for Kernelizing logistic regression.

----------------------
##### Perceptron | Cross-Entropy | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/Perceptron.py) 

##### Single layer perceptron is the foundation upon which Artificial Neural Networks (ANN) is built. We solved the optimization problem of the single layer perceptron using the 'Perceptron Convergence algorithm'.
----------------------
##### Perceptron | Convergence algo | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/Perceptron_stepwise.py)

##### The loss function here is solved using the 'Stochastic gradient descent' algorihm. Note that 'Perceptron Convergence algorrithm' is also a variant of the stochastic gradient descent algorithm except it uses the stepwise activation function.
----------------------
##### Linear Regression (Closed form) | Gradient descent | Stochastic GD | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Regression.py)

##### Linear regression is a foremost statistical model for predictive learning. Here we solved the regression problem by deriving its closed form solution. Since, the closed form solution is computationally expensive for large datasets, especially because the complexity of solving the matrix inverse of $X^TX$ is $O(N^{3})$, Researchers developed a stochastic gradient descent algorithm for regression. 
----------------------

##### Support Vector Machine (SVM) | [code](https://github.com/algostatml/SUPERVISED-ML/tree/master/REGRESSION/SupportVectorMachine(SVM))
##### [linearSVM code](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/SupportVectorMachine(SVM)/svm.py) | [kernelSVM](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/SupportVectorMachine(SVM)/kernelSVM.py)

##### Maximum margin classifier as it is commonly referred is a powerful discriminative classification algorithm. Its objective is to find data points that best seperates the data into distinctive classes. Using the data points the SVM algorithm established the optimumm margin for classification. SVM is also able to accurately classify non-linear data using the kernel trick.
----------------------
