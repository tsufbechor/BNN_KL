
Comparison of KL divergence values of Bayesian neural networks models trained on different variations of MNIST
KL Divergence:
=

KL(P∥Q) is a measure of the information gained by revising one's beliefs from the prior probability 
distribution Q to the posterior probability distribution P. 
A posterior probability is the probability of assigning observations to groups given the data. A prior 
probability is the probability that an observation will fall into a group before you collect the data.
Random labels.

Exercise:
=
I will compare the KL divergence values of Bayesian neural networks models trained on MNIST with and without the labelrandomization process.
Without randomization:
=
i.A classifier trained on the full dataset.
ii. A classifier trained on the first 200 samples of MNIST.
iii. A classifier trained on the 200 first ‘3’ and ‘8’ samples.
iv. A classifier on all ‘3’ and ‘8’ samples.

With randomization:
=
Using the first 200 samples of MNIST, I generated a random label for each sample from Bernoulli distribution
