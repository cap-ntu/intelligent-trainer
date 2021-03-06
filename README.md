# Train the Trainer!
![CAP](https://user-images.githubusercontent.com/9161548/40165577-eff023c4-59ee-11e8-8bf5-508325a23baa.png)
##### Developed by [CAP](http://cap.scse.ntu.edu.sg/)
The project aims to reduce the algorithm tweak cost for Model-based DRL (MBDRL). 
The approach is to package the standrad MBDRL training process into a RL environment TPE, and design an online trained intelligent trainer to control the sampling and training process inside TPE.

To cite our paper:
```bash
@misc{1805.09496,
Author = {Yuanlong Li and Linsen Dong and Yonggang Wen and Kyle Guan},
Title = {Intelligent Trainer for Model-Based Reinforcement Learning},
Year = {2018},
Eprint = {arXiv:1805.09496},
}
```

## 1. How to use

### 1.1 Install everything
We use `Python3.5` and [Anaconda](https://www.anaconda.com/download/) to manage the packages, 
the required packages are listed at the file `package-list.txt`. 
So firstly, install `Anaconda 3.6 version`  if you don't have one. 

1.1.1 First, clone this repository to your local PC.

```bash
git clone https://Lukeeeeee@bitbucket.org/RLinRL/intelligenttrainerpublic.git
```

1.1.2 Build a new anaconda environment and install the packages:

```bash
cd path/to/intelligenttrainerpublic
conda env create -f package-list.txt
source activate intelligenttrainer
```

1.1.2 Install [mujoco 131](http://www.mujoco.org/) and [Mujoco-py](https://github.com/openai/mujoco-py).

For mujoco, you should install [mjpro 131](https://www.roboti.us/index.html), and also register a [license](https://www.roboti.us/license.html). 

For mujoco-py, firstly follow mujoco-py [github page](https://github.com/openai/mujoco-py#install-mujoco) 
to do some configurations on your mjpro131 and license,

Then install mujoco-py by running:
```bash
source activate intelligenttrainer
pip install mujoco-py==0.5.7
```

1.1.3 Install [OpenAI Gym](https://github.com/openai/gym/) and [OpenAI baselines](https://github.com/openai/baselines)

We use specific version of gym which can support mjpro 131 and mujoco-py 0.5.7 due to compatibility problem

Install the openai gym by:
```bash
source activate intelligenttrainer
git clone https://github.com/openai/gym
cd gym
git checkout 1d8565717206e54fca9e73ea1b3969948b464c3c
pip install -e .
```

Install openai baselines by following its [github page](https://github.com/openai/baselines#installation).


### 1.2 Run different experiments
Firstly activate the anaconda environment:
```bash
source activate intelligenttrainer
```

#### 1.2.1 Run the Single-Head Trainer experiments
With branch "master", by running testIntelligent.py one can test the trainer framework with various single-head trainer implementations.  Inpendent experiments of multiple times with same configuration are uspported.
 (If you stopped the test before it fully finished, be sure to kill the process in your system.
If not, some errors within tensorflow and cuda may appear)

#### 1.2.2 Run the ensemble trainer experiments
With another branch "EnsembleTrainer", By running testRandomEnsemble.py, one can test the ensemble trainer.

#### 1.2.3 Results and Visualize:

Every time you run test, the log file will be stored automatically in the `log/` directory.

`log/baselineTestLog` : the baseline experiments log 

`log/intelligentTestLog`: the intelligent trainer experiments log.

For each directory, it stored in each test cases sub directory 
and then named by the time you start to run the experiments.

Like `/log/baselineTestLog/MountainCarContinuous-v0/2018-05-14_17-15-13IDSTRING` 
means a baseline trainer experiments running on environment `MountainCarContinuous-v0` 
with time stamp `2018-05-14_17-15-13` and speficication string `IDSTRING'

For each sub directory, it was structured in this way:
```bash
log/.../2018-05-14_17-15-13/config: the configuration file you used in this experiments.
log/.../2018-05-14_17-15-13/loss: record all training process information, like loss, reward etc.
```

For visualization, please check plotter.py in src/util.


## 2. Design of codes
### 2.1 A very simple UML schema about the desgin of our code.
![main UML](https://user-images.githubusercontent.com/9161548/40037703-37ea380e-5841-11e8-99f2-f760608e34b5.png)

### 2.1 Some important classes
2.1.1 The source code was structured in this way: 
1.	`src/` stores all the source code
2.	`log/` stores all the log file of each experiments
3.	`config/` stores all the configuration files like hyper-parameters of neural networks we used during experiments
4.	`test/` stores all experiments test files

2.1.2
There are some important classes:

a. `Class Agent (/src/agent/agent.py)`, is the entity for representing an agent in reinforcement learning problem, 
which is a very common design. Some important methods and attributes are listed below:

i.	`Agent.sample()`: sample some samples from a certain environment

ii.	`Agent.predict()`: get an action from its own model by pass into an observation, the model can be any policies, 
like DQN, DDPG or random policy. We design this by using strategy pattern in Design Pattern.

iii.`Agent.status`: An attribute stores agents current status, test or train. 

b.	`Class TargetAgent(/src/agent/targetAgent/targetAgent.py)`: is the entity for our target agent in our framework.
 It inherited from the Class Agent. Some important methods and attributes are listed below:

i.	`TargetAgent.train()`: train its own model, like a DDPG or TRPO

ii.	`TargetAgent.env_status`: a status representing the target agent is sampling from real environment or cyber 
environment, we use a setter method to modify this status. The agent's memory and other environment related attributes
 will be switched automatically.
 
iii.`TargetAgent.predict()`: We add the epsilon-greedy and action noise in this method

c.	`Class IntelligentTrainerAgent Class IntelligentRandomTrainerAgent Class BaselineTrainerAgent`

All three classes are the entities of our trainer, some design and methods are similar to Class TargetAgent, 
since all of them are used in a reinforcement learning problem formulation.

f.	`Class BaiscEnv (/src/env/env.py)`: an abstract class for environments, which also inherited from OpenAI gym's 
env class. Our cyber environment, training process environment are inherited from BasicEnv

i.	`BasicEnv.step()`: get an action and compute the state transition.

g.	`Class BaselineTrainerEnv(/src/env/trainerEnv/baselineTrainerEnv.py)` is the class where we define the baseline 
training process and abstract it into an environment.

i.	`BaselineTrainerEnv.step()`: define one step in training process, which include: target agent samples from real
 environment, cyber environment, and train the cyber environment mode, train the target agent by using real samples and cyber samples, test the cyber environments model and target agent.

h.	`Class TrainerEnv(/src/env/trainerEnv/trainerEnv.py)`: we define the training environment which inherited from 
BaselineTrainerEnv, and we generate the observation and reward which will return to intelligent trainer 
agent in this class.

2.	`Class Model(/src/model/model.py)`: We design this class to represent any policy or reinforcement learning 
algorithms in here based on the strategy design pattern. We implemented the DDPG, DQN, REINFORCE and TRPO here.
 Also we implement some basic policy, like a policy with fixed output which is used in baseline trainer agent, 
 the model that dynamics environment used which is a multi-layer neural network.

a.	`Model.predict()`: Get the output of the model by passing into the input.

b.	`Model.update()`: Update the model using its own method

3.	`Class Sampler (src/util/sampler/sampler.py)`: We derived the sampling function from agent to this class, 
so all agent's (including target agent, trainer agent) sampling utility is implemented by its won sampler. 

a.	`Sampler.sample()`: the function an agent will call when it wants to sample certain amount of samples 
from a environment

4.	`Class IntelligentSampler (src/util/sampler/intelligentSampler.py)`: a special sampler inherited from sampler
 which we used in intelligent trainer agent. It implemented our intelligent reset utility in here.

a.	`IntelligentSampler.reset()`: this reset override the `Class Sampler.sample()`, by using a quality function 
to select a good initial state.

