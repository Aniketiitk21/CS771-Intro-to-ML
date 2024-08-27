import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
    # Transform challenges to higher-dimensional feature vectors
    train_features = my_map(X_train)
    
    # Train linear models for Response0 and Response1 using Logistic Regression
    model0 = LogisticRegression(max_iter=10000).fit(train_features, y0_train)
    model1 = LogisticRegression(max_iter=10000).fit(train_features, y1_train)
    
    # Extract weights and bias terms
    w0, b0 = model0.coef_[0], model0.intercept_[0]
    w1, b1 = model1.coef_[0], model1.intercept_[0]
    
    return w0, b0, w1, b1


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points
    features = []
    for c in X:
        single_bits = c
        pairwise_products = np.array([c[i] * c[j] for i in range(len(c)) for j in range(i + 1, len(c))])
        features.append(np.concatenate((single_bits, pairwise_products)))
    feat=np.array(features)    
    return feat
