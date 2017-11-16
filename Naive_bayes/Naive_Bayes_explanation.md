# Bayes theory
## Bayesian decision theory
we can view the label y as the event A, and the x as the event B. As we all know that It's the A that caused B. So if y = {c1, c2, c3,...cN}, the P(ci|x) we called posterior probability. the λij is called the expected loss of category the x into ci, it will be defined as follows:

R(ci|x) = ∑ λij * P(cj|x)(j equals from 1 to N)

So for every x, if the ci = h(x), so that R(h(x)|x) is the smallest for each x. Then the R(h) = Ex[R(h(x)|x)] is the smallest.

When each λij is 1 if i != j, while λij = 0 if i == j(the target of us is minimize the mistaking rate), then:

R(ci|x) = ∑P(cj|x)(j is 1 to N)
        = 1 - P(ci|x)

Then you only need to minimize the P(c|x).
Here is the question, how can we get the P(c|x)?
### Calculate the P(c|x)
#### 1. Directly create P(c|x)

The decision tree, BP NeuralNetwork and Support Machine Vector are using this method.

#### 2.Calculate the P(x, c) and P(x)

 Depending on the bayes theory, P(x, c) = P(c) * P(x|c), the P(x|c) is the class-conditional probability or we can say likelihood. We can use Law of Large Numbers to calculate the P(c), but for P(x|c), the frequency of the sample category is so large that even the examples can't cover them.

 (1)calculate the P(x|c) using probability distribution

this way we calculate class-conditional probablity is to assume that it has a certain probability distribution,and it's decided by a parameter theta(θ), then we can represent the P(x|c) as P(x|θc)

Then how to calculate the θ, Frequentist usually use Maximum Likelihood Estimation to calculate. The Bayesian think θ is also a random variance, having probability distribution. So we can assume the parameter has a prior distribution, then use the data to calculate the posterior distribution.

The Maximum Likelihood Estimation like this

P(Dc|θc) = Π P(x|θc)   //x belong to Dc, we know that every x is indepently distributed

Using log-likelihood method the result is like this:

 θc = argmax LL(θc) = argmax log P(Dc|θc) = argmax ∑log P(x|θc)

 (2)Without knowing probability distribution to calculate it.

 We also use the Maximum Likelihood Estimation for each feature.

 P(x|c) = Π P(xi|c)  // i is from 1 to d and d is the number of features. Here we assume that all the features are indepently distributed

 ## Naive Bayes classifier
We find that for all the category, the P(x) is the same(We randomly choose the sample, if for every feature, it is binary, then the combination is 2^n, and every p(x) is 1/(2^n)) so the expression of naive bayes classifier is:

h(x) = argmaxP(c)Π(xi|c)  //i is one to d

P(c) = |Dc|/|D| //D is the set

P(xi|c) = |Dc,xi|/|Dc| // Dc,xi is the set of whose ith feature is xi in Dc. **This is just for discrete attribute**

For continuous attribute, if the feature obbey Normal distribution,then we can use the formula.

But if one value of certain feature doesn't show in the dataset, multiply by it will cause the probability equals to 0. So we often use the smoothing method and Laplacian correction.

P(c) = (|Dc| + 1)/(|D| + N)

P(xi|c) = (|Dc,xi|+1)/(|Dc| + Ni)

N is the number of labels and Ni is the number of ith feature value.




