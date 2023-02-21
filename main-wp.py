import tensorflow_probability
import tensorflow
import numpy

tfd = tensorflow_probability.distributions
initialDistribution = tfd.Categorical(probs=[0.8, 0.2])
transitionDistribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observationDistribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(initialDistribution, transitionDistribution, observationDistribution, num_steps=7)

mean = model.mean()
with tensorflow.compat.v1.Session() as sess:
  print(mean.numpy())
