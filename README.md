
Comparison of KL divergence values of Bayesian neural networks models trained on different variations of MNIST
KL Divergence:
=

KL(P∥Q) is a measure of the information gained by revising one's beliefs from the prior probability 
distribution Q to the posterior probability distribution P. 
A posterior probability is the probability of assigning observations to groups given the data. A prior 
probability is the probability that an observation will fall into a group before you collect the data.
Random labels.
Bayesian Neural Network:
=
Bayesian neural network (BNN) combines neural network with Bayesian inference. Simply speaking, in BNN, we treat the weights and outputs as the variables and we are finding their marginal distributions that best fit the data

Exercise:
=
I will compare the KL divergence values of Bayesian neural networks models trained on MNIST with and without the labelrandomization process.

Without randomization:
=
i. A classifier trained on the full dataset.
ii. A classifier trained on the first 200 samples of MNIST.
iii. A classifier trained on the 200 first ‘3’ and ‘8’ samples.
iv. A classifier on all ‘3’ and ‘8’ samples.

With randomization:
=
Using the first 200 samples of MNIST, I generated a random label for each sample from Bernoulli distribution

Results:
=
![image](https://user-images.githubusercontent.com/81694762/222192891-d313297d-a8a5-4341-9bde-a2250f3b31a1.png)

Conclusions:
=
An important conclusion from this exercise is the fact that decrease in KL divergence from epoch to 
epoch is very strongly correlated to the number of training examples the model was trained on. The 
steepest decrease was found in the first model trained on all examples, followed by the model 
trained on only 2s and 8s.
Intuitively, it makes sense that the distance between P and Q (Posterior and Prior) are a function of 
the number of samples the model is trained on but I must explore this topic further to determine if 
this is universally true.
