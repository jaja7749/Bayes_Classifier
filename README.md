# Bayes optimal classifier
A Bayes classifier, also known as a Naive Bayes classifier, is a probabilistic machine learning algorithm based on Bayes' theorem. It is a simple and efficient classification algorithm that is particularly well-suited for high-dimensional datasets. The "naive" part of its name comes from the assumption of independence between features, meaning that the presence or absence of a particular feature is assumed to be unrelated to the presence or absence of other features given the class label.

Here's a basic overview of how a Bayes classifier works:

1. Bayes' Theorem:
   - The algorithm is based on Bayes' theorem, which is a mathematical formula that describes the probability of an event, based on prior knowledge of conditions that might be related to the event. In the context of classification, it helps us calculate the probability of a class given some observed features.
     
   - Bayes' theorem is expressed as follows:
     
     - $`P(class|features)= \frac{P(features|class)\cdot P(class)}{P(features)}`$
       
     - Where:
       - $`P(class|features)`$ is the posterior probability of the class given the features.
       - $`P(features∣class)`$ is the likelihood of observing the features given the class.
       - $`P(class)`$ is the prior probability of the class.
       - $`P(features)`$ is the probability of observing the features.

2. Naive Assumption:
   - The "naive" assumption in Naive Bayes comes from assuming that the features are conditionally independent given the class label. This simplifying assumption makes the computations more tractable, especially for high-dimensional datasets.

3. Training:
   - During the training phase, the algorithm learns the prior probabilities $`P(class)`$ and the likelihoods
   - $`P(features∣class)`$ from the training data.

4. Classification:
   - When a new instance with features is presented for classification, the algorithm calculates the posterior probabilities for each class using Bayes' theorem. The class with the highest posterior probability is then assigned as the predicted class for the input.

5. Types of Naive Bayes Classifiers:
   - There are different variants of Naive Bayes classifiers, such as:
     - Gaussian Naive Bayes: Assumes that the features follow a Gaussian (normal) distribution.
     - Multinomial Naive Bayes: Suitable for discrete data, often used for text classification.
     - Bernoulli Naive Bayes: Appropriate for binary feature data.

Naive Bayes classifiers are known for their simplicity, speed, and efficiency, especially in situations with limited training data. They are commonly used in applications like text classification, spam filtering, and sentiment analysis.


# Naive Bayes Classifier on E-mail Classification
- The email data file is on: https://github.com/justmarkham/DAT5/blob/master/data/SMSSpamCollection.txt

|   spam   |    ham    |
|----------|-----------|
|     1    |     -1    |

|  Index  |   Label   |                      Content                       |
|---------|-----------|----------------------------------------------------|
|    0    |    -1     |   Go until jurong point, crazy.. Available only ...|
|    1    |    -1     |                       Ok lar... Joking wif u oni...|
|    2    |     1     |   Free entry in 2 a wkly comp to win FA Cup fina...|
|    3    |    -1     |   U dun say so early hor... U c already then say...|
|    4    |    -1     |   Nah I don't think he goes to usf, he lives aro...|

First of all, we create a class to input $`X`$ and $`y(label)`$:
```ruby
class NBC:
    def __init__(self, X, y, col):
        self.X = np.array(X)
        self.y = np.array(y) #label
        self.col = col #classes
        self.p_y = np.zeros(2)
        self.p_x_y = np.zeros((2, len(self.X.T)))
```

Train the data, we need to calculate $`P(y+)`$, $`P(y-)`$, $`P(x|y+)`$ and $`P(x|y-)`$ to model:
```ruby
    def train(self):
        self.p_y[0] = np.sum(self.y==+1)/len(self.y) #p(y+)
        self.p_y[1] = np.sum(self.y==-1)/len(self.y) #p(y-)
        for d in range(len(self.X.T)):
            self.p_x_y[0][d] = (np.sum(self.X.T[d][y==+1]>=1)+1) / (np.sum(self.X.T[d][y==+1]>=0)+2) #p(x|y+)
            self.p_x_y[1][d] = (np.sum(self.X.T[d][y==-1]>=1)+1) / (np.sum(self.X.T[d][y==-1]>=0)+2) #p(x|y-)
```

After we have $`P(y+)`$, $`P(y-)`$, $`P(x|y+)`$ and $`P(x|y-)`$, the model can detect the e-mail is spam or ham:
```ruby
    def pred(self, x):
        y_pred = np.array([])
        for d in range(1 if x.ndim == 1 else len(x)):
            p_X_y = np.ones(2)
            p_x_y = abs(1-(x[d]>=1)-self.p_x_y)
            for i in range(len(p_x_y[0])):
                p_X_y *= p_x_y[:,i] #p(X|y) = Πp(x|y)
            p_x = p_X_y[0]*self.p_y[0]+p_X_y[1]*self.p_y[1] #p(x)
            y_pred = np.append(y_pred, 1 if (p_X_y[0]*self.p_y[0]/p_x >= p_X_y[1]*self.p_y[1]/p_x) else -1) #p(y|x) = p(x|y)*p(y)/p(x)
        return y_pred
```

Now, let's try to vaild the dataset:
```ruby
model = NBC(X, y, col)
model.train()
y_pred = model.pred(X)
#print(f"Accuracy: {model.valid()*100:.3f} %")
print(f"Accuracy: {(np.sum(y==y_pred)/len(y))*100:.3f} %")
cm = confusion_matrix(y, y_pred, labels=[-1, 1])
cm_display = ConfusionMatrixDisplay(cm, display_labels=[-1, 1])
cm_display.plot()
plt.show()
```
![image](https://github.com/jaja7749/Bayes_Classifier/blob/main/images/Naive%20Bayes%20confusion%20matrix.png)

And test the new data:
```ruby
# test1, test2 are [spam, ham]
test1 = "You are a winner U have been specially selected 2 receive £1000 cash or a 4* holiday (flights inc) speak to a live operator 2 claim 0871277810810"
test2 = "I just reached home. I go bathe first. But my sis using net tell u when she finishes k..."
test = np.array([test1, test2])
X_ = dataset.datatransform(data=test, new_data=True)
print("Naive Bayes Predict(1:spam, -1:ham):", model.pred(X_))

# Naive Bayes Predict(1:spam, -1:ham): [ 1. -1.]
```

# Gaussian Naive Bayes on Different Distribution Data
we create the dataset from sklearn:
```ruby
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons, make_gaussian_quantiles

samples = 200
datasets = [
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=1),
    make_blobs(n_samples=samples, centers=2, n_features=2, random_state=6),
    make_moons(n_samples=samples, noise=0.15, random_state=0),
    make_circles(n_samples=samples, noise=0.15, factor=0.3, random_state=0),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=2, random_state=0),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=1, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2, n_clusters_per_class=1),
    make_classification(n_samples=samples, n_features=2, random_state=1, n_redundant=0, n_informative=2),
    make_blobs(n_samples=samples, centers=3, n_features=2, random_state=1),
    make_blobs(n_samples=samples, centers=4, n_features=2, random_state=6),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=3, random_state=0),
    make_gaussian_quantiles(n_samples=samples, n_features=2, n_classes=4, random_state=0),
]
```
<img src="https://github.com/jaja7749/Bayes_Classifier/blob/main/images/different%20distribution.png" width="720">

On Gaussian (Normal) distribution, we can find the model parameter from Maximum Likehood Estimation (MLE).

- Naive Bayes:

$`P(X|y)=\prod_{d=1}^{D}P(x_{d}|y)`$

- Gaussian (Normal) distribution:

$`P(x_d|y)=\frac{1}{\sigma \sqrt{2\pi }}e^{-\frac{1}{2}(\frac{x-\mu }{\sigma })^{2}}`$

- Maximum Likehood Estimation (MLE):

$`L(x^{(i)})=\prod_{i=1}^{N}\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}(\frac{x^{(i)}-\mu}{\sigma})^{2}}\left [ \left | y^{(i)}=k \right | \right ]`$

$`\Rightarrow \ln L(x^{(i)})=\sum_{i=1}^{N}(-\ln \sigma-\frac{1}{2}(\frac{x^{(i)}-\mu}{\sigma^{2}})^{2}+\ln\frac{1}{\sqrt{2\pi}})\left [ \left | y^{(i)}=k \right | \right ]`$

1. Average: $`\frac{\partial\ln L}{\partial\mu}=0\ \Rightarrow \ \mu=\frac{1}{N}\sum_{i=1}^{N}x^{(i)}`$
2. Variance: $`\frac{\partial\ln L}{\partial\sigma}=0\ \Rightarrow \ \sigma^{2}=\frac{1}{N}\sum_{i=1}^{N}(x^{(i)}-\mu)^{2}`$

First of all, we create a class to input $`X`$ and $`y(label)`$:
```ruby
class GNB:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.label, self.p_y = np.unique(y, return_counts=True)
        self.p_y =  self.p_y.astype(float) / np.sum(self.p_y)
        self.avg = np.zeros((len(self.X.T), len(self.label)))
        self.var = np.zeros((len(self.X.T), len(self.label)))
```

Train the data, we need to calculate $`Average`$ and $`Variance`$ to model:
```ruby
    def train(self):
        for i in range(len(self.X.T)):
            for j in range(len(self.label)):
                self.avg[i][j] = np.average(self.X[:,i][self.y==self.label[j]]) #µ
                self.var[i][j] = np.var(self.X[:,i][self.y==self.label[j]]) #σ^2
```

After we have $`Average`$ and $`Variance`$, the model can classified data:
```ruby
    def probability(self, x, avg, var):
        return (1/np.sqrt(2*np.pi*var))*np.exp(-0.5*np.power(x-avg, 2)/var)
        
    def pred(self, x):
        y_pred = np.array([])
        for d in range(1 if x.ndim == 1 else len(x)):
            p_X_y = np.ones(len(self.label))
            p_y_X = np.ones(len(self.label))
            for i in range(len(self.label)):
                for j in range(len(self.X.T)):
                    p_X_y[i] *= self.probability(x[j] if x.ndim == 1 else x[d][j], self.avg[j][i], self.var[j][i])
            p_X = np.sum(p_X_y*self.p_y)
            p_y_X = p_X_y*self.p_y/p_X
            y_pred = np.append(y_pred, self.label[np.argmax(p_y_X)])
        return y_pred
```

Now, let's try to vaild the dataset:
```ruby
col = 2
row = int(len(datasets)/col)
plt.subplots(row, col, figsize=(12, 24))
for i in range(col):
    for j in range(row):
        X = datasets[row*i+j][0]
        y = datasets[row*i+j][1]
        
        plt.subplot(row, col, row*i+j+1)
        model = GNB(X, y)
        model.train()
        plt.title(f"Accuracy: {model.valid()*100:.3f} %")
        model.plot()
```

<img src="https://github.com/jaja7749/Bayes_Classifier/blob/main/images/Gaussian%20Naive%20Bayes%20result.png" width="720">

# (Multivariate) Gaussian Bayes (Quadratic Discriminant Analysis, QDA) on Different Distribution Data

Dataset is same as Gaussian Naive Bayes.

On Multivariate Gaussian (Normal) distribution, we can find the model parameter from Maximum Likehood Estimation (MLE).

- Bayes (without Naive Assumption):

$`P(x|y)`$

- Gaussian (Normal) distribution:

$`P(x|y)={\displaystyle (2\pi )^{-k/2}\det({\boldsymbol {\Sigma }})^{-1/2}e^{ \left(-{\frac {1}{2}}(\mathbf {x} -{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}(\mathbf {x} -{\boldsymbol {\mu }})\right)}}`$

- Maximum Likehood Estimation (MLE):

1. Average: $`\frac{\partial\ln L}{\partial\mu}=0\ \Rightarrow \ \mu=\frac{1}{N}\sum_{i=1}^{N}x^{(i)}`$
2. Covariance: $`\frac{\partial\ln L}{\partial\Sigma}=0\ \Rightarrow \ \Sigma=\frac{1}{N}\sum_{i=1}^{N}(x^{(i)}-\mu)(x^{(i)}-\mu)^{T}`$

First of all, we create a class to input $`X`$ and $`y(label)`$:
```ruby
class MGB:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.label, self.p_y = np.unique(y, return_counts=True)
        self.p_y =  self.p_y.astype(float) / np.sum(self.p_y)
        self.avg = np.zeros((len(self.label), len(self.X.T)))
        self.cov = np.zeros((len(self.label), len(self.X.T), len(self.X.T)))
```

Train the data, we need to calculate $`Average`$ and $`Covariance`$ to model:
```ruby
    def train(self):
        for i in range(len(self.label)):
            for j in range(len(self.X.T)):
                self.avg[i][j] = np.average(self.X[:,j][self.y==self.label[i]]) #µ
                self.cov[i] = np.cov(self.X[self.y==self.label[i]].T) #Σ
```

After we have $`Average`$ and $`Covariance`$, the model can classified data:
```ruby
    def probability(self, x, avg, cov):
        return (1/np.power(2*np.pi, len(self.label)/2))*(1/np.power(np.linalg.det(cov), 1/2))*np.exp((-1/2)*np.dot(np.dot((x-avg).T, np.linalg.inv(cov)), (x-avg)))
        
    def pred(self, x):
        y_pred = np.array([])
        for d in range(1 if x.ndim == 1 else len(x)):
            p_X_y = np.ones(len(self.label))
            p_y_X = np.ones(len(self.label))
            for i in range(len(self.label)):
                p_X_y[i] = self.probability(x if x.ndim == 1 else x[d], self.avg[i], self.cov[i])
            p_X = np.sum(p_X_y*self.p_y)
            p_y_X = p_X_y*self.p_y/p_X
            y_pred = np.append(y_pred, self.label[np.argmax(p_y_X)])
        return y_pred
```

Now, let's try to vaild the dataset:
```ruby
col = 2
row = int(len(datasets)/col)
plt.subplots(row, col, figsize=(12, 24))
for i in range(col):
    for j in range(row):
        X = datasets[row*i+j][0]
        y = datasets[row*i+j][1]
        
        plt.subplot(row, col, row*i+j+1)
        model = MGB(X, y)
        model.train()
        plt.title(f"Accuracy: {model.valid()*100:.3f} %")
        model.plot()
```

<img src="https://github.com/jaja7749/Bayes_Classifier/blob/main/images/Gaussian%20Bayes%20result.png" width="720">
