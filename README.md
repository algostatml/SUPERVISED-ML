# SUPERVISED-ML
Supervised machine learning algorithms explained with code

----------------------
##### Logistic Regression | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/LogisticRegression.py)

##### Here we present the solution to logistic regression using the cross-entropy loss function. This is a binary classifier for supervised classification problem and we solve the optimization function derived from the maximum likelihood using gradient ascent.
----------------------
##### Perceptron | Cross-Entropy [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/Perceptron.py) | 

##### Single layer perceptron is the foundation upon which Artificial Neural Networks (ANN) is built. We solved the optimization problem of the single layer perceptron using the 'Perceptron Convergence algorithm'.
----------------------
##### Perceptron | Convergence algo [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/CLASSIFICATION/Perceptron_stepwise.py)

##### The loss function here is solved using the 'Stochastic gradient descent' algorihm. Note that 'Perceptron Convergence algorrithm' is also a variant of the stochastic gradient descent algorithm except it uses the stepwise activation function.
----------------------
##### Linear Regression (Closed form) | Gradient descent | Stochastic GD | [code](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/Regression.py)

##### Linear regression is a foremost statistical model for predictive learning. Here we solved the regression problem by deriving its closed form solution. Since, the closed form solution is computationally expensive for large datasets, especially because the complexity of solving the matrix inverse of $X^TX$ is $O(N^{3})$, Researchers developed a stochastic gradient descent algorithm for regression. 
----------------------

----------------------
##### Support Vector Machine (SVM) | [linearSVM code](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/SupportVectorMachine(SVM)/svm.py) | [kernelSVM](https://github.com/algostatml/SUPERVISED-ML/blob/master/REGRESSION/SupportVectorMachine(SVM)/kernelSVM.py)

##### Maximum margin classifier as it is commonly referred is a powerful discriminative classification algorithm. Its objective is to find data points that best seperates the data into distinctive classes. Using the data points the SVM algorithm established the optimumm margin for classification. SVM is also able to accurately classify non-linear data using the kernel trick.
----------------------
