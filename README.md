 GA Deep Network

Consider a deep network to learn the AI actions for a character in a game (Enemies, Knife, Gun. Health, Run, Hide, Wander, Attack)

What are the parameters that we can experiments with?


Number of layers

Nodes per layer

Learning Rate

Initial weights standard deviation

Loss Function

Optimizer

what else????


Consider combining efforts with other teams.   For example, one team may want to write the GA.  Another team may want to edit/modify the TensorFlow program to communicate with the GA either through files, pipes, sockets, batch files, etc.  You do not need to write everything in python.

Create a genetic algorithm that evolves the parameters for a deep neural network.  We want to optimize the efficiency of the neural network.

Analysis:

Discuss your Chromosomes.  What makes up your chromosomes?

How did you measure fitness?  Ultimately we want a accurate neural network that trains quickly.  You can consider your fitness measurement to be the number of epochs required to train the test data accurately.

What happens if your training data trains accurately but your test data is not 100%?

Experiment with different values of N, Pc, Pm?   What are your findings?

In this study, N may need to be low because the fitness measurement will take a while.  It could take 5 to 10 seconds to run approximately 1000 epochs.