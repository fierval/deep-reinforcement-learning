
# Deep Reinforcement Learning Nanodegree: Finished Projects

This repository contains solutions to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program projects

## Projects

- [x] [Navigation](p1_navigation): In the first project, you will train an agent to collect yellow bananas while avoiding blue bananas. Here is the [Report](p1_navigation/Report.md)
- [ ] [Continuous Control](p2_continuous-control): In the second project, you will train an robotic arm to reach target locations.
- [ ] [Collaboration and Competition](p3_collab-compet): In the third project, you will train a pair of agents to play tennis! 

## Setup

### GPU Support (optional)

I am working with a Titan V GPU on Windows and Ubuntu. Highly recommended for deep learning, though not required.

Becuse of this, I have removed `tensorflow 1.7.1` from `requirements.txt` used to install other requirements (see below). If you have GPU support and Tensorflow becomes required:

```bash
pip install -U --ignore-installed tensorflow-gpu==1.7.1
```

### PyTorch

`requirements.txt` (see below) lists `torch==0.4.0`. This step fails on Windows with Anaconda. I have removed this line from `requirements.txt` and separated PyTorch installation into its own step.

### Conda Environment

To set up your python environment to run the code in this repository, follow the instructions below. Environment created will contain all of conda packages. If not desired omit "anaconda" at the end of each line.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6 anaconda
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 anaconda
	activate drlnd
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).

3. Install pytorch. Go to [https://pytorch.org/](https://pytorch.org/) and scroll down to the installation instructions for your OS.
	
4. Install Unity ml-agents in the environment. From repository root:
   ```bash
    cd python
	pip install -r requirements.txt
    ```
5. Launch `jupyter notebook` from the active environment (after running `activate drlnd`).