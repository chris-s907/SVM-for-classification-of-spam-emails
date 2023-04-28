## Introduction
The Spam Data Set saved in the matrix with 57 attributes of a feature vector. We apply hard-margin SVM of linear kernel, hard-margin SVM of polynomial kernel and soft-margin of polynomial kernel. We calculated the discriminant function and get the optimal hyperplane of each case. Then we calculated the training and testing accuracy of each case with different hyperparameters and do comparison. 

We also use the RBF SVM model to this dataset and form the evaluation set to assess the performance of RBF SVM model. We calculate the training and evaluation accuracy of different hyperparameters of RBF method and do analysis.

## Data pre-processing
We standardize the data by removing the mean value of each feature and then dividing by each feature’s standard deviation. We normalize the training and testing data with the equation

## Compute discriminant function
 * Admissibility of the kernels
 We compute the Gram matrix with Mercer’s condition and calculate all the eigenvalues of the matrix. The matrix K contains some very small negative values. We set a very small negative value -{10}^{-4} as the threshold. As long as there is no eigenvalues smaller than it, then we believe the matrix is positive semi-definite and the kernel candidate is admissible which ensures that the SVM optimization problem is convex and has a unique global minimum. 
 
 * A hard-margin SVM with the linear kernel
 * A hard-margin SVM with the polynomial kernel
 * A soft-margin SVM with the polynomial kernel
## Training and testing accuracy calculation
 * A hard-margin SVM with the linear kernel	
 * A hard-margin SVM with the polynomial kernel
 * A soft-margin SVM with the polynomial kernel
## Radial Basis Function (RBF) kernel and implementation
