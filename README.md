# Machine-Learning
This repo contains all of the Machine Learning Algorithms implementations from the Machine Learning course of Coursera taught by **Andrew Ng** and offered by Stanford University.

The course is similar to the **CS229** course by **Stanford University** available on its website and YouTube channel.

Coursera:
> https://www.coursera.org/learn/machine-learning

YouTube:
> https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599


# NOTES
## GRADIENT DESCENT (GD):
In gradient descent as we approach a local minimum, gradient descent will automatically take smaller steps. And that's because by definition the local minimum is when the derivative is equal to zero & as we approach local minimum, this derivative term will automatically get smaller, and so gradient descent will automatically take smaller steps. That's why there is no need to decrease alpha or the time.

### TRICKS TO MAKE GRADIENT DESCENT FASTER:
- **Feature Scaling**: Try to make all the different features to take similar range of values. Because if there is a large difference in the ranges of different features, the contours will be moreover very tall and skinny ellipses, due to which gradient descent will oscillate back and forth and will take much longer time to reach the global minimum.<br>
But when there is feature scaling there will be similar ranges for different features which will form more circle like contours for the cost function i.e. there will be a moreover direct path to reach to the global minimum.
- One more technique to make Gradient Descent faster is **MEAN NORMALIZATION**. In this method the values of all training examples of that particular feature is updated such that the mean of those is equal to zero.

To implement both of these techniques, adjust your input values as shown in this formula:
xi:= (xi − μi)/si
Where μi is the average of all the values for feature (i) and si is the range of values (max - min), or si is the standard deviation.
**Note that dividing by the range, or dividing by the standard deviation, give different results.**


### Working of GRADIENT DESCENT: (alpha = Learning Rate)
- **A large value of alpha can lead to overshooting** (increase in the value of cost function after an iteration) instead of reaching to the global minimum which is only possible when cost function strictly decreases with each iteration.
- For sufficiently small alpha, cost function should decrease on every iteration. But **if alpha is too small, gradient descent can be slow to converge.**


### Comparison of Gradient Descent and Normal Equation:
#### Gradient Descent                           Normal Equation
- Need to decide appropriate alpha.         - No need to decide alpha.
- Needs many iterations.                    - Don't need to iterate.
- Works well even when n is large.          - Need to compute inverse of (x'x), that's why slow if n is large.<br>
(Time Complexity to compute inverse of (x'x)= O(n^3))<br>
**To summarize, so long as the number of features is not too large, the normal equation gives us a great alternative method to solve for the parameter theta.**


### Normal Equation Non-Invertibility:
Reasons for inverse of (x'x) to be non invertible -
- Redundant features (Linearly Dependent): Remove all other linearly dependent features and keep only one.
- Too many features (n >= m): Remove some features, or use regularization.


## Underfitting and Overfitting:
- **Underfitting**, or **high bias**, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features.
- At the other extreme, **overfitting**, or **high variance**, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:
1) Reduce the number of features:
- Manually select which features to keep.
- Use a model selection algorithm.
2) Regularization
- Keep all the features, but reduce the magnitude of parameters theta_j.
- Regularization works well when we have a lot of slightly useful features.


J = (h - y)^2 + lamda * theta^2
The λ, or lambda, is the regularization parameter. It determines how much the costs of our theta parameters are inflated.

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting.


Regularized Linear Regression:
theta = theta * (1 - alpha * lambda / m) - alpha * ((h - y) * x) / m
Normal Equation-
To add in regularization, the equation is the same as our original, except that we add another term:
theta = inverse of(x'x + lambda * L) * x'y
L = Identity matrix except 0 at (1,1)
It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x(0)), multiplied with a single real number lambda.
Recall that if m < n, then X'X is non-invertible. However, when we add the term λ⋅L, then X'X + λ⋅L becomes invertible.


Neural Networks:
Each layer gets its own matrix of weights, theta^j.
The dimensions of these matrices of weights is determined as follows:
If network has s_(j) units in layer j and s_(j+1) units in layer j+1, then theta^j will be of dimension s_(j+1) x s_(j)+1.
The +1 comes from the addition in theta^(j) of the "bias nodes", x_0 and theta^j_(0). In other words the output nodes will not include the bias nodes while the inputs will.


Model Representation:
We can then add a bias unit (equal to 1) to layer j after we have computed a^j. This will be element a^j_(0).
1. To compute our final hypothesis, let's first compute another z vector:
z^(j+1) = Θ^(j) * a^(j)
We get this final z vector by multiplying the next theta matrix after Θ^(j−1) with the values of all the activation nodes we just got. This last theta matrix Θ^(j) will have only one row which is multiplied by one column a^j so that our result is a single number. We then get our final result with:
hΘ(x)=a^(j+1) = g(z^(j+1))
Notice that in this last step, between layer j and layer j+1, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.


Putting Neural Networks all computations together:

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.
-Number of input units = dimension of features x^i
-Number of output units = number of classes
-Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
-Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

Training a Neural Network
-Randomly initialize the weights
-Implement forward propagation to get hΘ(x^i) for any x^i
-Implement the cost function
-Implement backpropagation to compute partial derivatives
-Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
-Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.

When we perform forward and back propagation, we loop on every training example.


Learning Curves:
Experiencing high bias:
Low training set size: causes Jtrain(Θ) to be low and JCV(Θ) to be high.
Large training set size: causes both Jtrain(Θ) and JCV(Θ) to be high with Jtrain(Θ)≈JCV(Θ).
If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.

Experiencing high variance:
Low training set size: Jtrain(Θ) will be low and JCV(Θ) will be high.
Large training set size: Jtrain(Θ) increases with training set size and JCV(Θ) continues to decrease without leveling off. Also, Jtrain(Θ) < JCV(Θ) but the difference between them remains significant.
If a learning algorithm is suffering from high variance, getting more training data is likely to help.


Bias, Variance and Diagnosis:
Our decision process can be broken down as follows:
1) Getting more training examples: Fixes high variance
2) Trying smaller sets of features: Fixes high variance
3) Adding features: Fixes high bias
4) Adding polynomial features: Fixes high bias
5) Decreasing λ: Fixes high bias
6) Increasing λ: Fixes high variance

Diagnosing Neural Networks
-A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
-A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.

Model Complexity Effects:
1) Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
2) Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
3) In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.


Building Spam Classifier:
So how could you spend your time to improve the accuracy of this classifier?
1) Collect lots of data (for example "honeypot" project but doesn't always work)
2) Develop sophisticated features (for example: using email header data in spam emails)
3) Develop algorithms to process your input in different ways (recognizing misspellings in spam).
It is difficult to tell which of the options will be most helpful.


Error Analysis:
The recommended approach to solving machine learning problems is to:
1) Start with a simple algorithm, implement it quickly, and test it early on your cross validation data.
2) Plot learning curves to decide if more data, more features, etc. are likely to help.
3) Manually examine the errors on examples in the cross validation set and try to spot a trend where most of the errors were made.
