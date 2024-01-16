# Bayes optimal classifier
A Bayes classifier, also known as a Naive Bayes classifier, is a probabilistic machine learning algorithm based on Bayes' theorem. It is a simple and efficient classification algorithm that is particularly well-suited for high-dimensional datasets. The "naive" part of its name comes from the assumption of independence between features, meaning that the presence or absence of a particular feature is assumed to be unrelated to the presence or absence of other features given the class label.

Here's a basic overview of how a Bayes classifier works:

## Bayes' Theorem:
The algorithm is based on Bayes' theorem, which is a mathematical formula that describes the probability of an event, based on prior knowledge of conditions that might be related to the event. In the context of classification, it helps us calculate the probability of a class given some observed features.

Bayes' theorem is expressed as follows:

$`P(class|features)= \frac{P(features|class)\cdot P(class)}{P(features)}`$
​

Where:
- $`P(class|features)`$ is the posterior probability of the class given the features.

- $`P(features∣class)`$ is the likelihood of observing the features given the class.

- $`P(class)`$ is the prior probability of the class.

- $`P(features)`$ is the probability of observing the features.

# Naive Assumption:
The "naive" assumption in Naive Bayes comes from assuming that the features are conditionally independent given the class label. This simplifying assumption makes the computations more tractable, especially for high-dimensional datasets.

# Training:
During the training phase, the algorithm learns the prior probabilities $`P(class)`$ and the likelihoods $`P(features∣class)`$ from the training data.

# Classification:
When a new instance with features is presented for classification, the algorithm calculates the posterior probabilities for each class using Bayes' theorem. The class with the highest posterior probability is then assigned as the predicted class for the input.

# Types of Naive Bayes Classifiers:
There are different variants of Naive Bayes classifiers, such as:

- Gaussian Naive Bayes: Assumes that the features follow a Gaussian (normal) distribution.
- Multinomial Naive Bayes: Suitable for discrete data, often used for text classification.
- Bernoulli Naive Bayes: Appropriate for binary feature data.

Naive Bayes classifiers are known for their simplicity, speed, and efficiency, especially in situations with limited training data. They are commonly used in applications like text classification, spam filtering, and sentiment analysis.
