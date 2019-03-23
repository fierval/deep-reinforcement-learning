[//]: # (Image References)

[mu_s]: http://bit.ly/2TvjiNH "mu s"
[qu_s_mu_s]: http://bit.ly/2TrSNs7 "qu s mu s"
[qu_s_a]: http://bit.ly/2TsbXhJ "qu s a"

# Solving the Reacher Environment with DDPG

## Learning algorithm

Deep Deterministic Policy Gradient belongs to the Actor Critic class of the policy-based learning algorithms genus, although it would probably be more precise to call it a "Master-Slave" method (after the famous chapter in Phenomenology by G. W. F. Hegel), after all, actor, while minimizing its loss, is working to minimize that of critic's.

Just like in the descrete DQN space (or any RL space), we are working to maximize the ![qu s a][qu_s_a] funcion. If ![mus s][mu_s] is what we use to get our actions, and everything is deterministic, we can substitute ![mu s][mu_s] into the value function directly and obrain the gradient by chain rule: ![qu s mu s][qu_s_mu_s] is going to then be a neural net that accepts the output of actor neural net together with the action! Very neat.