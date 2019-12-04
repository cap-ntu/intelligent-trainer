### Exp log
2019 12 04
1. Pen car swimmer halfcheetah no dyna rerun
2. Pen car swimmer halfcheetah RANDOM rerun
3. Pen car swimmer halfcheetah DQN rerun

2018.8.15

1. Update the reset function called it 'V3'

Run baseline on 6 cases with action to 0.4, 0.5, 0.6, 0.7, 0.8 ####
But bug occurs on Inverted Pendulum cases

2. Decrease the real sample count to 10%, and run intel on 5 cases based on V3
2.1 Swimmer, HalfCheetah, Reacher, MountainCarContinuous #### 
2.2 InvertedPendulum error due to change of config(dyna max sample horizon)

3. For target that use TRPO, set the dynamics horizon to the same as real env, 
since it is Monte Carlo based.
Case: Reacher, Swimmer, HalfCheetah, InvertedPendulum

# TODO
3.1 Baseline 0.4, 0.5, 0.6, 0.7, 0.8 
3.1.1 Baseline 0.7 But with name 0.6 on log dir #####
3.1.2 Baseline 0.5 running ####

3.2
3.2.1 Intelligent 
Still running 
3.2.2 Intelligent 10 Percent 
still running

8.16
4. Update the new step function called `v4`
4.1 Run on intel v4  still running





2018.5.12 
27. Assemble on 5 cases (intel v2, no dyna, random intel v2):
    27.1 Pen ####
    27.2 Car ####
    27.3 Swimmer 
    27.4 Half
    27.5 Reacher 
    
2018.5.11:
25. All tricks (direct agent reward, split action 5, reinforce 5 on intel v2):
    25.1 Car server #####
    25.2 Pen server ####
    25.3 Swimmer 
    25.4 Half
    25.5 Reacher ######
    
26. Redo intel v2 on swimmer reacher 
    26.1 Swimmer 10  sets
    26.2 reacher 10 sets

2018.5.7
21. Action split to [0.2 0.4 0.6 0.8 1.0] on Intel v1 ##########
    21.1 Car server!!!
    21.2 Pen server!!!
    21.3 Swimmer
    21.4 Half
    21.5 Reacher 
    
22. Action split to [0.2 0.4 0.6 0.8 1.0] on Intel v2 ###########
    22.1 Car server!!!
    22.2 Pen server!!!
    22.3 Swimmer
    22.4 Half
    22.5 Reacher 
23. Action split to [0.2 0.4 0.6 0.8 1.0] on REINFORCE ###########
    23.1 Car step size 20, 40 
    23.2 Swimmer step 1
    23.3 Pen step size 1
    23.4 Reacher 1
    23.5 Half Cheetah 1

24. Swimmer Intel v2 
    24.1 Reward use target agent reward directly ########
    24.2 F1 set to 0 constant. ########

2018.5.6
13. Random intel v1 0.8 tmux: random_intel_v1_0_8_seed_2 new seed!!! #############
    13.1 pen 10 sets
    13.2 car 10 sets
14.  Random intel v1 0.9 tmux: random_intel_v1_0_9 #########
    14.1 pen 10 sets
    14.2 car 10 sets

15. REINFORCE step 1 
    15.1 pen 10 sets #########
    15.2 car 10 sets #########
    15.3 reacher 10 sets ######
    15.4 half 10 sets #########
    15.5 swimmer 10 sets #########

16. REINFORCE step 5 
    16.1 pen 10 sets  #########
    16.2 car 10 sets #########
    16.3 reacher 10 sets #######
    16.4 half 10 sets #######
    16.5 swimmer 10 sets ########
    
17. REINFORCE step 10
    17.1 pen 10 sets #########
    17.2 car 10 sets #######

18. REINFORCE step 20 #########
    18.1 pen 10 sets ########
    18.2 car 10 sets ########

19. REINFORCE step 5 lr 0.01 #########
    19.1 pen 10 sets
    19.2 car 10 sets
20. Intel v2 with prediction on reward AR method ##########
    20.1 pen 10 sets
    20.2 car 10 sets

2018-05-05
7. Random intel : CAP server tmux: random_intel_swim_rec_half######### 
    5.1 Reacher 10 sets !!!!! #########
    5.2 Swimmer 10 sets !!!!! #############
    5.3 HalfCheetah 10 sets !!!!!  #########
8. Random intel  0.8 tmux: random_intel_v1_0_8 ############
    6.1 pen 10 sets
    6.2 car 10 sets
9. Random intel  0.3 tmux: random_intel_v1_0_3 ############
    7.1 pen 10 sets
    7.2 car 10 sets
10. Randonm intel  with only one action dimension tmux: random_dim_1_intel ############
    8.1 pen 10 sets
    8.2 car 10 sets
11. Random intel  no reset ! tmux: random_no_set ############
    9.1 pen 10 sets
    9.2 car 10 sets
12. half log: test log of intelligent trainer action tmux : half_log ############
    10.1 halfCheetah 1 sets


2018-05-04
1. Pendulum(done)######################
    change to tanh, and re-run all the experiments: scse server
    1.1. no dyanamics 10 sets !!!!!!!!! 
    1.2 baseline 0.6 .0.6 10 sets !!!!!!!!!
    1.3 random 10 sets !!!!!!!!
    1.3 intel v1 10 sets !!!!!!!!
    1.4 intel v2 10 sets !!!!!!!!
 
2. MountainCarContinuous(done)######################
    change to tanh, and re-run all the experiments: scse server
    2.1. no dyanamics 10 sets !!!!!!!!!
    2.2 baseline 0.6 .0.6 10 sets !!!!!!!!!
    2.3 random 10 sets !!!!!!!
    2.3 intel v1 10 sets !!!!!!!!
    2.4 intel v2 10 sets !!!!!!!
  
3. Baseline no dynamics: CAP server tmux : baseline_no_dyna
    3.1 Reacher 10 sets !!!!!! 
    3.2 Swimmer 10 sets !!!!!! done !!!! ######################
    3.3 HalfCheetah !!!!!!  done !!!!######################
    
4. dqn tuning CAP server tmux : tun_dqn (done) ######################
    4.1 change state to sample count ration, change reward to 2rd order difference of target agent real sample reward
        4.1.1 Pen 10 sets
        4.1.2 Car 10 sets
    4.2 change state to sample count ration and cyber loss, change reward to 2rd order difference of target agent real sample reward 
        4.2.1 Pen 10 sets
        4.2.2 Car 10 sets
        
5. baseline no dynamics tmux: baseline_no_dyna
    11.1 reacher 10 sets  ######
    11.2 swimmer 10 sets ############
    11.3 halfCheetah 10 sets #############
6. intel v0 (we did not run reach swimmer halfcheetah for intel v0 before, so fix with it)  ######################
    12.1 reacher 8 sets ######
    12.2 swimmer 5 sets #######
    12.3 halfCheetah 7 sets #######   
