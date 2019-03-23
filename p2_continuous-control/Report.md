[//]: # (Image References)

[mu_s]: http://bit.ly/2CCobi2 "mu_s"

# Solving the Reacher Environment with DDPG

## Learning algorithm

Deep Deterministic Policy Gradient belongs to the Actor Critic class of the policy-based learning algorithms genus, although it would probably be more precise to call it a "Master-Slave" method (after the famous chapter in Phenomenology by G. W. F. Hegel), after all, actor, while minimizing its loss, is working to minimize that of critic's.

Just like in the descrete DQN space (or any RL space), we are working to maximize the `Q(s, a)` funcion. If ![mus_s][mu_s]