---
title: Machine Learning Note - cs229 - Stanford
date: 2017-12-05
author: "Cong Chan"
tags: ['Machine Learning']
---
参考
[CS229: Machine Learning, Stanford](http://cs229.stanford.edu/notes)

什么是机器学习？目前有两个定义。

亚瑟·塞缪尔（Arthur Samuel）将其描述为：“不需要通过具体的编程，使计算机能够学习”。这是一个较老的，非正式的定义。

汤姆·米切尔（Tom Mitchell）提供了一个更现代的定义：
E：经验，即历史的数据集。
T：某类任务。
P：任务的绩效衡量。
若该计算机程序通过利用经验E在任务T上获得了性能P的改善，则称该程序对E进行了学习
“如果计算机程序能够利用经验E，提升实现任务T的成绩P，则可以认为这个计算机程序能够从经验E中学习任务T”。
例如：玩跳棋。E =玩许多棋子游戏的经验，T = 玩跳棋的任务。P = 程序将赢得下一场比赛的概率。
<!-- more -->

## [Supervised Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
### Linear Regression
* Weights(parameters) θ: parameterizing the space of linear functions mapping from X to Y
* Intercept term: to simplify notation, introduce the convention of letting x<sub>0</sub> = 1
* Cost function J(θ): ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/linearR_cost.png)  a function that measures, for each value of the θ’s, how close the h(x<sup>(i)</sup>)’s are to the corresponding y<sup>(i)</sup>’s
* Purpose: to choose θ so as to minimize J(θ).
* Implementation: By using a search algorithm that starts with some “initial guess” for θ, and that repeatedly changes θ to make J(θ) smaller, until hopefully we converge to a value of θ that minimizes J(θ).

#### LMS(least mean squares) algorithm:
* gradient descent
* learning rate
* error term
* batch gradient descent：looks at every example in the entire training set on every step
* stochastic gradient descent(incremental gradient descent)：repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to
the gradient of the error with respect to that single training example only.
* particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

#### The normal equations
performing the minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize J by explicitly taking its derivatives with respect to the θ<sub>j</sub>’s, and setting them to zero.
To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let’s introduce some notation for doing calculus with matrices
* Matrix derivatives: the gradient ∇<sub>A</sub>f(A) is itself an m-by-n matrix, whose (i, j)-element is ∂f/∂A<sub>ij</sub>
* Least squares revisited: Given a training set,
   * define the design matrix X to be the m-by-n matrix (actually m-by-n + 1, if we include the intercept term) that contains the training examples’ input values in its rows,
   * let y be the m-dimensional vector containing all the target values from the training set,
   * used the fact that the trace of a real number is just the real number( trace operator, written “tr.” For an n-by-n matrix A, the trace of A is defined to be the sum of its diagonal entries: trA = ΣA<sub>ii</sub>
   * To minimize J, find its derivatives with respect to θ: ∇<sub>θ</sub>J(θ) = X<sup>T</sup>Xθ − X<sup>T</sup>y
   * To minimize J, we set its derivatives to zero, and obtain the normal equations: X<sup>T</sup>Xθ = X<sup>T</sup>y
   * Thus the value of θ that minimizes J(θ) is given in closed form by the equation: θ = (X<sup>T</sup>X)<sup>-1</sup>X<sup>T</sup>y

#### Probabilistic interpretation
why the least-squares cost function J is a reasonable choice? With a set a probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

#### Locally weighted linear regression (LWR) algorithm
assuming there is sufficient training data, makes the choice of features less critical.
* In the original linear regression algorithm, to make a prediction at a query point x (i.e., to evaluate h(x)), we would:
  1. Fit θ to minimize Σ<sub>i</sub>(y<sup>(i)</sup> − θ<sup>T</sup>x<sup>(i)</sup>)<sup>2</sup>.
  2. Output θ<sup>T</sup>x.
* The locally weighted linear regression algorithm does the following:
  1. Fit θ to minimize Σ<sub>i</sub>w<sup>(i)</sup>(y<sup>(i)</sup> − θ<sup>T</sup>x<sup>(i)</sup>)<sup>2</sup>.
  2. Output θ<sup>T</sup>x.
* Here, the w<sup>(i)</sup>’s are non-negative valued **weights**
* Intuitively, if w<sup>(i)</sup> is large for a particular value of i, then in picking θ, we’ll try hard to make (y<sup>(i)</sup> − θ<sup>T</sup>x<sup>(i)</sup>)<sup>2</sup> small. If w<sup>(i)</sup> is small, then the error term will be pretty much ignored in the fit.
* A fairly standard choice for the weights is w<sup>(i)</sup> = exp(-(x<sup>(i)</sup>-x)<sup>2</sup> / 2τ<sup>2</sup> )
* if |x<sup>(i)</sup>-x| is small, then w<sup>(i)</sup> ≈ 1; if large, then w<sup>(i)</sup> is small. Hence, θ is chosen giving a much higher “weight” to the (errors on) training examples close to the query point x.
* The parameter τ controls how quickly the weight of a training example falls off with distance of its x<sup>(i)</sup>, from the query point x; τ is called the **bandwidth** parameter

### Classification and logistic regression
#### Logistic regression
* logistic function or the **sigmoid function**: g(z) = (1 + e<sup>−z</sup>)<sup>-1</sup>. ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/lr.png)
g(z) tends towards 1 as z → ∞, and g(z) tends towards 0 as z → −∞. ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/sigmoid.png)
* derivative of the sigmoid function:  g(z)<sup>'</sup> = g(z)(1 - g(z))
* endow our classification model with a set of probabilistic assumptions, and then fit the parameters via maximum likelihood:
  * Similar to our derivation in the case of linear regression, we can use gradient ascent to maximize the likelihood.
  * updates will therefore be given by θ := θ + α∇<sub>θ</sub>ℓ(θ). (Note the positive rather than negative sign in the update formula, since we’re maximizing,rather than minimizing, a function now.)
  * This therefore gives us the stochastic gradient ascent rule: θ<sub>j</sub> := θ<sub>j</sub> + α(y<sup>(i)</sup>− h<sub>θ</sub>(x<sup>(i)</sup>))x<sup>(i)</sup><sub>j</sub>
  * If we compare this to the LMS update rule, we see that it looks identical; but this is not the same algorithm, because h<sub>θ</sub>(x<sup>(i)</sup>) is now defined as a non-linear function of θ<sup>T</sup>x<sup>(i)</sup>.
     * Nonetheless, it’s a little surprising that we end up with the same update rule for a rather different algorithm and learning problem. Is this coincidence, or is there a deeper reason behind this? Check [GLM models](#generalized-linear-models).  

### Generalized Linear Models
#### The exponential family
* Bernoulli distributions
* Gaussianexponential distributions
* multinomial
* Poisson (for modelling count-data)
* beta and the Dirichlet (for distributions over probabilities)

#### Constructing GLMs
* Ordinary Least Squares
* Logistic Regression
* Softmax Regression

##### Softmax Regression
Consider a classification problem in which the response variable y can take on any one of k values, so y ∈ {1, 2, . . . , k}. We will thus model it as distributed according to a multinomial distribution.
* parameterize the multinomial with only k − 1 parameters, φ<sub>1</sub>, . . . , φ<sub>k−1</sub>, where φ<sub>i</sub> = p(y = i; φ), and p(y = k; φ) = 1 − Σ<sup>k</sup><sub>i=1</sub>φ<sub>i</sub>.
* To express the multinomial as an exponential family distribution, we will definee T(y) ∈ R<sup>k-1</sup>：
![T(y)](/image/cs229-notes1-pic01.png)
  * η = [log(φ<sub>1</sub>/φ<sub>k</sub>),...,log(φ<sub>k-1</sub>/φ<sub>k</sub>)], the η<sub>i</sub>’s are linearly related to the x’s.
  * softmax function: a mapping from the η’s to the φ’s: φ<sub>i</sub> = e<sup>ηi</sup> / Σ<sup>k</sup><sub>j=1</sub>e<sup>ηi</sup>
* softmax regression:  the model, which applies to classification problems where y ∈ {1, . . . , k}: p(y = i|x; θ) = φ<sub>i</sub> = e<sup>θ<sup>T</sup><sub>i</sub> x</sup> / Σ<sup>k</sup><sub>j=1</sub>e<sup>θ<sup>T</sup><sub>i</sub> x</sup>  
* This hypothesis will output the estimated probability that p(y = i|x; θ), for every value of i = 1, . . . , k.
* parameter fitting: obtain the maximum likelihood estimate of the parameters by maximizing ℓ(θ) in terms of θ, using a method such as gradient ascent or Newton’s method.

### Naive Bayes classification 朴素贝叶斯
以二元分类为例: 根据A和B各自的先验概率和条件概率, 算出针对某一特征事件的后验概率, 然后正则化(正则化后两个后验概率之和为1, 但不影响对事件的触发对象是A或B的判断)
* Why naïve: 忽略了事件发生的顺序, 故称之为"朴素"
* Strength and Weakness: 高效, 快速, 但对于组合性的短语词组, 当这些短语与其组成成分的字的意思不同时, NB的效果就不好了
* 详见[加速自然语言处理-朴素贝叶斯](https://github.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/blob/master/%E5%8A%A0%E9%80%9F%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86.md#naive-bayes-classifier)
* Problem: how to deal with continuous values features? Use Gaussian Naive Bayes.

#### Gaussian Naive Bayes
With real-valued inputs, we can calculate the mean and standard deviation of input values (x) for each class to summarize the distribution. This means that in addition to the probabilities for each class, we also store the mean μ and standard deviations σ of each feature for each class.
* The class conditional probability P(x|c) is estimated by probability density of the normal distribution ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/probability_density.png):
* Algorithm – continuous Xi (but still discrete Y)
    * Train Naïve Bayes (examples)
    ```
    for each class value yk:
        estimate P(Yk)
        for each attribute Xi:
            estimate class conditional mean, variance
    ```
    * Classify(xnew): ` Ynew <- argmax(k) ∏P(xi|Yk)P(Yk)`
* Short: classes with the same distribution

#### Missing data instances in NB
* Ignore attribute in instance where its value is missing
* compute likelihood based on observed attribtues
* no need to “fill in” or explicitly model missing values
* based on conditional independence between attributes

### Generative and Discriminative Algorithm:
* Generative classifiers learn a model of the joint probability, p(x, y), of the inputs x and the label y, and make their predictions by using Bayes rules to calculate p(y|x), and then picking the most likely label y.
* Discriminative classifiers model the posterior p(y|x) directly, or learn a direct map(hypothesis/functions) from inputs x to the class labels.
* Generative models advantage:
    * Can be good with missing data, naive Bayes handles missing data
    * good for detecting outliers
    * to generate likely input (x,y).

### Decision trees 决策树
* Algorithm: ID3 algorithm
* Decision trees with continuous attributes: Create split based on threshold

#### ID3 algorithm
Recursive Split( node, {examples} ):
    1. A <- the best attribute for splitting the {examples}
    2. For each value of A, create new child node
    3. Split training {examples} to child nodes
    4. For each child node, subset:
     * If subset is pure - stop
     * Else: split(child_node, {subset} )
How to decide which attribute is the best to split on: Entropy

#### Entropy
* ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/entropy.png)
* Use log2 here is to represent concepts of information - on average how many bits needed to tell X split purity
	* To represent two classes, need one bit "0, 1", to represent 4 classes, need 2 bits "00, 01, 10, 11"
    * If x is pure(one class only), entropy is 0.
* Information Gain: Expected drop in entropy after split, Gain( P, C) = Entropy(parent) - Σw*Entropy(children), w is weighted average matrix.![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/infogain.png), A is the split attribute
    * Problems: tend to pick attributes with lots of values, could not generalize well on new data.
	* use GainRation: for attribute A with many different values V, the SplitEntropy will be large, ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/SplitEntropy.png)

#### Overfitting in Decision Trees
the tree split too deep to try to classify almost every single sample. As a result the model could not predict new data well.
* Sub-tree replacement pruning
	1. For each node:
		* Pretend remove node + all children from the tree
		* Measure performance on validation set
	2. Remove node that results in greatest improvement
	3. Repeat until further pruning is harmful

#### Decision boundary
Logistic Regression and trees differ in the way that they generate decision boundaries
* Decision Trees bisect the space into smaller and smaller regions,
* Logistic Regression fits a single line/hyperplane to divide the space exactly into two.

#### Random Decision forest
* Grow K different decision trees:
	* pick a random subset Sr of training examples
	* grow a full ID3 tree (no prunning):
		* When splitting: pick from d<<D random attributes
		* Computing gain based on Sr instead of full set
	* repeat for r =1…K
* Given a new data point X:
	* classify X using each of the trees T1 …. Tk
	* use majority vote: class predicted most often


### SVM
* Intuition: Suppose there is a good hyperplane to seperate data set, h(x)=g(w<sup>T</sup>x+b), (relation with fully connected layer and activation funciton in DNN).
    * Want functional margin of hyperplane to be large: for dataset (xi,yi), functional margin γi = yi(w<sup>T</sup>xi+b), if yi=1, need w<sup>T</sup>xi+b>>0, if yi=-1, need w<sup>T</sup>xi+b<<0. Thus γi>0 means the classification is correct.
    * Geometric margins: Define the hyperplane as w<sup>T</sup>x+b=0, the normal of the hyperplane is w/||w||, thus a point A(xi)'s, which represents the input x(i) of some training example with label y(i) = 1, projection on the hyperplane is point B = xi - γi·w/||w||, where γi is xi's distance to the decision boundary. Thus w<sup>T</sup>(xi - γi·w/||w||) + b=0 => γi =  (w/||w||)<sup>T</sup>xi+ b/||w||. More generally, the geometric margin of (w, b) with respect to a training example (xi, yi) is γi = yi· (w/||w||)<sup>T</sup>xi+ b/||w||
    * If ||w|| = 1, then the functional margin equals the geometric margin
* The optimal margin classifier: Given a training set, a natural desideratum is to try to find a decision boundary that maximizes the minimum (geometric) margin, i.e want min(γi) as large as possible. Via some [transformation](http://cs229.stanford.edu/notes/cs229-notes3.pdf), the object turns to minimize ||w||<sup>2</sup>, subject to y(i)·(w<sup>T</sup>xi+b) ≥ 1,
* [Lagrange duality](http://cs229.stanford.edu/notes/cs229-notes3.pdf): solving constrained optimization problems. w = Σαiyixi, αi is Lagrange multipliers.
* Support vector: The points with the smallest margins. The number of support vectors can be much smaller than the size the training set
* Training: fit our model’s parameters to a training set, and now wish to make a prediction at a new point input x. We would then calculate w<sup>T</sup>x + b, and predict y = 1 if and only if this quantity is bigger than zero. ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/svm.png).
    * In order to make a prediction, we have to calculate it which depends only on the inner product between x and the points in the training set.
    * Moreover, αi’s will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we need to find only the inner products between x and the support vectors (of which there is often only a small number) in order to make our prediction.
    * The inner product <xi,x> could be replaced by kernel k(xi,x)

#### Kernels
* Define the “original” input value x as the input attributes of a problem. When that is mapped to some new set of quantities that are then passed to the learning algorithm, we’ll call those new quantities the input features.
* φ denote the feature mapping, which maps from the attributes to the features. E.g. φ(x) = [x, x^2, x^3]
* given a feature mapping φ, we define the corresponding Kernel to be K(x, z) = φ(x)<sup>T</sup>φ(z)
* Often, φ(x) itself may be very expensive to calculate (perhaps because it is an extremely high dimensional vector, require memory), K(x, z) may be very inexpensive to calculate.
* We can get SVMs to learn in the high dimensional feature space given by φ, but without ever having to explicitly find or represent vectors φ(x). E.g.![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/kernel1.png) ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/kernel2.png)
* Based on [Mercer’s Theorem](https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/lectures/lec3.pdf), you can either explicitly map the data with a φ and take the dot product, or you can take any kernel and use it right away, without knowing nor caring what φ looks like
* Keep in mind however that the idea of kernels has significantly broader applicability than SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products <x, z> between input attribute vectors, then by replacing this with K(x, z) where K is a kernel, you can allow your algorithm to work efficiently in the high dimensional feature space corresponding to K.

#### SVM vs. Logistic regression
* Logistic regression focuses on maximizing the probability of the data. The further the data lies from the separating hyperplane (on the correct side), the happier LR is.
* An SVM don’t care about getting the right probability, i.e the right P(y=1|x), but only care about P(y=1|x)/P(y=0|x)≥ c. It tries to find the separating hyperplane that maximizes the distance of the closest points to the margin (the support vectors). If a point is not a support vector, it doesn’t really matter.
* P(y=1|x)/P(y=0|x) > c, if c=1, that means P(y=1|x) > P(y=0|x), thus y=1, take log of both side, and plug in P(y=1|x) = sigmoid(w<sup>T</sup>x + b), P(y=0|x)=1-P(y=1|x), recall the [sigmoid](#logistic-regression), we get w<sup>T</sup>x + b > 0
* Underlying basic idea of linear prediction is the same, but error functions differ, the r = P(y=1|x)/P(y=0|x) = exp(w<sup>T</sup>x + b), different classifiers assigns different cost to r
    * If cost(r)=log(1 + 1/r), this is logistic regression
    * If cost(r)=max(0, 1-log(r))=max(0, 1-(w<sup>T</sup>x + b)), then SVM
    * Logistic regression (non-sparse) vs SVM (hinge loss, sparse solution)
    * Linear regression (squared error) vs SVM (ϵ insensitive error)

### K Nearest Neighbour
Intuition: predict based on nearby/similar training data.
* Algorithm: for a test data
    1. compute its distance to every training example xi
    2. select k closest training instances
    3. prediction:
        * For Classification: predict as the most frequent label among the k instances.
        * For regression: predict as the mean of label among the k instances.
* Choose k
    * large k: everything classified as the most probable class
    * small k: highly variable, unstable decision boundaries
    * affects “smoothness” of the boundary
    * Use train-validation to choose k
* Distance meansures:
    * Euclidian: symmetric, spherical, treats all dimensions equally, but sensitive to extreme differences in single attribtue
    * Hamming: number of attribtues that differ
* Resolve ties:
    * random
    * prior: pick class with greater prior
    * nearest: use 1-NN classifier to decide
* Missing values: have to fill in the missing values, otherwise cannot compute distance.
* Pro and cons:
    * Almost no assumptions about data
    * easy to update in online setting: just add new item to training set
    * Need to handle missing data: fill-in or create a special distance
    * Sensitive to outliers
    * Sensitve to lots of irrelevant aeributes (affect distance)
    * Computationally expensive: need to compute distance to all examples O(nd) - [Vectorization](http://cs229.stanford.edu/section/vec_demo/Vectorization_Section.pdf) ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/knn.png)
* Faster knn: K-D Trees, Inverted lists, Locality-sensitive hashing

#### K-D Trees
low-dimensional, real-valued data
* A kd-tree is a binary tree data structure for storing a finite set of points from a k-dimensional space.
* Build the tree: Pick random dimension, Find median, Split data
* Nearest neighbor search: Traverse the whole tree, BUT make two modifications to prune to search space:
    * Keep variable of closest point C found so far. Prune subtrees once their bounding boxes say that they can’t contain any point closer than C
    *  Search the subtrees in order that maximizes the chance for pruning

#### Inverted lists
high-dimensional, discrete data, sparse
* Application: text classification, most attribute values are zero (sparseness),
* training: list all training examples that contain particular attribute
* Testing: merge inverted list for attribtues presented in the test set, and choose those instances in the new inverted list as the neighbours

#### Locality-sensitive hashing
high-d, discrete or real-valued



## Unsupervised learning 无监督学习
### Clustering
#### K-means
split data into a specified number of populations
* Input: 
    * K (number of clusters in the data)
    * Training set {x1, x2, x3 ..., xn) 
* Algorithm:
    * Randomly initialize K cluster centroids as {μ1, μ2, μ3 ... μK}, now centroid could represent cluster.
    * Repeat until converge:
        * Inner loop 1: repeatedly sets the c(i) variable to be the index of the closes variable of cluster centroid closes to xi, i.e. take ith example, measure squared distance to each cluster centroid, assign c(i)to the  closest cluster(centroid)
        * Inner loop 2: For each cluster j, new centroid c(j) = average mean of all the points assigned to the cluster j in previous step.
* Target (Distortion) function: J(c,μ)=Σ|| xi-μi ||^2, coordinate ascent, decrease monotonically, thus guarantee to converge.
* What if there's a centroid with no data:
    * Remove that centroid, so end up with K-1 classes,
    * Or, randomly reinitialize it, not sure when though...
* How to choose cluster numbers: scree plot to find the best k.

#### Hierarchical K-means
* A Top-down approach
    1. run k-means algorithm on the original dataset
    2. for each of the resulting clusters, recursively run k-means
* Pro cons:
    * Fast
    * nearby points may end up in different clusters

#### Agglomerative Clustering
A bottom up algorithm:
```
1. starts with a collections of singleton clusters
2. repeat until only one cluster is left:
    1. Find a pair of clusters that is closest
    2. Merge the pair of clusters into one new cluster
    3. Remove the old pair of clusters
```
* Need to define a distance metric over clusters
![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/cluster_dist.png)
* Produce a dendrogram: Hierarchical tree of clusters
![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/Dendrogram.png)
* slow

#### Gaussian Mixtures
For non-Gaussian distribution data, assume it is a mixture of several(k) Gaussians.
* Algorithm: EM

#### EM algorithm
strategy will be to repeatedly construct a lower-bound on ℓ(E-step) based on [Jensen’s inequality](http://cs229.stanford.edu/notes/cs229-notes8.pdf), and then optimize that lower-bound(M-step).
* E step: For each i, let Qi be some distribution over the z’s (Σ<sub>z</sub>Qi(z) = 1, Qi(z) ≥ 0). z(i) indicating which of the k Gaussians each x(i) had come from, get P(Z)=φ, then compute the conditional probability wj as P(x|Z) via [Gaussian Naive Bayes](#gaussian-naive-bayes): ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/Estep.png)
* M step: maximize, with respect to our parameters φ, µ, Σ, the quantity![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/Mstep1.png),
    by updating parameter(φ, µ, σ) ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/Mstep2.png)
* 举例：start with two randomly placed Gaussians (μa, σa), (μb, σb), assume a uniform prior (P(a)=P(b)=0.5), iterate until convergence:
    * E-step: for each point: P(b|xi), P(a|xi)=1-P(b|xi) ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/EM_e.png), does it look like it came from b or a?
    * M-step: adjust (μa, σa) and (μb, σb) to fit points **soft** assigned to them, ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/EM_m1.png)
    ![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/EM_m2.png)
* The EM-algorithm is also reminiscent of the K-means clustering algorithm, except that instead of the “hard” cluster assignments c, we instead have the “soft” assignments w. Similar to K-means, it is also susceptible to local optima, so reinitializing at several different initial parameters may be a good idea.
* How to pick k: cannot discover K, likelihood keeps growing with K

#### K-means vs. EM
![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/km&em.jpg)

### Dimensionality Reduction
* Pros:
    * reflects human intuitions about the data
    * allows estimating probabilities in highadimensional data: no need to assume independence etc.
    * dramatic reduction in size of data: faster processing (as long as reduction is fast), smaller storage
* Cons
    * too expensive for many applications (Twitter, web)
    * disastrous for tasks with fine-grained classes
    * understand assumptions behind the methods (linearity etc.): there may be better ways to deal with sparseness

#### Factor analysis
If the features n ≫ m, or n≈m, in such a problem, it might be difficult to model the data even with a single Gaussian, 更别提高斯混合了. Because the variance matrix Σ becomes singular - [non invertable](http://cs229.stanford.edu/notes/cs229-notes9.pdf).
####  Principal Components Analysis
PCA, automatically detect and reduce data to lower dimension k, k << n, preserve dimenson that affects class separability most.
* Algorithm:
    * Pre-process: data normalization to 0 mean and unit variance![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/pca_norm.png),
    Steps (3-4) may be omitted if we had apriori knowledge that the different attributes are all on the same scale
    * to project data into a k-dimensional subspace (k < n), we should choose e1,... ek to be the top k eigenvectors of Σ. The e’s now form a new, orthogonal basis for the data.
    * To represent a training data point x with d dimension into this basis (k dimension), e1<sup>T</sup>x,...ek<sup>T</sup>x
* The vectors u1,..., uk are called the first k principal components of the data.
* Eigenvalue λi = variance along ei.
    * Pick ei that explain the most variance by sorting eigenvectors s.t. λ1 ≥ λ2 ≥…≥ λn
    * pick first k eigenvectors which explain 90% or 95% of the total variance Σλ(i).
* Maximize the variance of projection of x onto a unit vector u,
* Application: eigenfaces

#### Linear Discriminant Analysis
LDA
* Idea: pick a new dimension that gives
    * maximum separation between means of projected classes
    * minimum variance within each projected class
* How: eigenvectors based on between-class and within-class covariance matrices
* LDA not guaranteed to be better for Classification
    * assumes classes are unimodal Gaussians
    * fails when discriminatory information is not in the mean, but in the variance of the data

#### Singular Value Decomposition

## Generalization and evaluation
### Receiver Operating Characteristic
ROC, plot TPR(Sensitivity) vs. FPR(Specificity) as t varies from ∞ to -∞, shows performance of system across all possible thresholds
![](https://raw.githubusercontent.com/ShootingSpace/Computer-Science-and-Artificial-Intelligence/master/image/roc.png)
* A test with perfect discrimination (no overlap in the two distributions) has a ROC curve that passes through the upper left corner. Therefore the closer the ROC curve is to the upper left corner, the higher the overall accuracy of the test
* AUC: area under ROC curve, popular alternative to Accuracy

### Confidence interval
tell us how closed our estimation
* E = probability that misclassify a random instance: Take a random set of n instances, how many misclassified? Equal to Binomial distribution with mean = nE, variance = nE(1-E)
* Efuture: the next instance's probability of misclassified = average #misclassifed = variance / n = mean E=  E(1-E)/n, small variance means big confidence interval, a Gaussian distribution with one variance distance extend from mean will cover 2/3 future test sets
* p% Confidence interval for future error, 95% confidence interval needs about 2 variance extends from mean.

.
