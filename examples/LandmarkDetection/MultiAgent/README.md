# Anatomical Landmark Detection

Deep Reinforcement Learning (DRL) has proven to achieve state-of-the-art accuracy in medical imaging analysis. DRL methods can be leveraged to automatically find anatomical landmarks in 3D scanned images. Robust and fast landmark localisation is critical in multiple medical imaging analysis applications such as biometric measurements of anatomical structures, registration of 3D volumes and extraction of 2D clinical standard planes. Here, we explore more advanced approaches involving multiple cooperating agents with a focus on their communication in order to improve performances. The increase in accuracy could lead to a general adoption in clinical settings to reduce costs and human errors.  Our results show that the CommNet architecture with communicating agents on a single landmark outperforms previous approaches. We can detect the anterior commissure landmark with an average distance error of 0.75mm. Our implementations also have greater accuracy than expert clinicians on the apex and mitral valve centre.


<p align="center">
<img style="float: center;" src="images/cmarl_arch.png">
<br/>
<img style="float: center;" src="images/environment.png">
</p>

---
## Results
Here is an example of learned agents for landmark detection on unseen data:

<p align="center">
<img src="./videos/brain_5_agents.gif">
</p>


---

### Train
```
python DQN.py --task train --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 1 2 --model_name Network3d
```

### Evaluate
```
python DQN.py --task eval --load 'data/models/BrainMRI/network3d_5_agents.pt' --files 'data/filenames/image_files.txt' 'data/filenames/landmark_files.txt' --file_type brain --landmarks 13 14 0 1 2 --model_name "Network3d"
```

## Usage
```
usage: DQN.py [-h] [--load LOAD] [--task {play,eval,train}]
              [--file_type {brain,cardiac,fetal}] [--files FILES [FILES ...]]
              [--val_files VAL_FILES [VAL_FILES ...]] [--saveGif]
              [--saveVideo] [--logDir LOGDIR]
              [--landmarks [LANDMARKS [LANDMARKS ...]]]
              [--model_name {CommNet,Network3d}] [--batch_size BATCH_SIZE]
              [--memory_size MEMORY_SIZE]
              [--init_memory_size INIT_MEMORY_SIZE]
              [--max_episodes MAX_EPISODES]
              [--steps_per_episode STEPS_PER_EPISODE]
              [--target_update_freq TARGET_UPDATE_FREQ]
              [--save_freq SAVE_FREQ] [--delta DELTA] [--viz VIZ]
              [--multiscale] [--write] [--train_freq TRAIN_FREQ]

optional arguments:
  -h, --help            show this help message and exit
  --load LOAD           Path to the model to load (default: None)
  --task {play,eval,train}
                        task to perform, must load a pretrained model if task
                        is "play" or "eval" (default: train)
  --file_type {brain,cardiac,fetal}
                        Type of the training and validation files (default:
                        train)
  --files FILES [FILES ...]
                        Filepath to the text file that contains list of
                        images. Each line of this file is a full path to an
                        image scan. For (task == train or eval) there should
                        be two input files ['images', 'landmarks'] (default:
                        None)
  --val_files VAL_FILES [VAL_FILES ...]
                        Filepath to the text file that contains list of
                        validation images. Each line of this file is a full
                        path to an image scan. For (task == train or eval)
                        there should be two input files ['images',
                        'landmarks'] (default: None)
  --saveGif             Save gif image of the game (default: False)
  --saveVideo           Save video of the game (default: False)
  --logDir LOGDIR       Store logs in this directory during training (default:
                        runs)
  --landmarks [LANDMARKS [LANDMARKS ...]]
                        Landmarks to use in the images (default: [1])
  --model_name {CommNet,Network3d}
                        Models implemented are: Network3d, CommNet (default:
                        CommNet)
  --batch_size BATCH_SIZE
                        Size of each batch (default: 64)
  --memory_size MEMORY_SIZE
                        Number of transitions stored in exp replay buffer. If
                        too much is allocated training may abruptly stop.
                        (default: 100000.0)
  --init_memory_size INIT_MEMORY_SIZE
                        Number of transitions stored in exp replay before
                        training (default: 30000.0)
  --max_episodes MAX_EPISODES
                        "Number of episodes to train for" (default: 100000.0)
  --steps_per_episode STEPS_PER_EPISODE
                        Maximum steps per episode (default: 200)
  --target_update_freq TARGET_UPDATE_FREQ
                        Number of epochs between each target network update
                        (default: 10)
  --save_freq SAVE_FREQ
                        Saves network every save_freq steps (default: 1000)
  --delta DELTA         Amount to decreases epsilon each episode, for the
                        epsilon-greedy policy (default: 0.0001)
  --viz VIZ             Size of the window, None for no visualisation
                        (default: 0.01)
  --multiscale          Reduces size of voxel around the agent when it
                        oscillates (default: False)
  --write               Saves the training logs (default: False)
  --train_freq TRAIN_FREQ
                        Number of agent steps between each training step on
                        one mini-batch (default: 1)
```

## Citation

If you use this code in your research, please cite these papers:

```
@article{leroy2020communicative,
  title={Communicative Reinforcement Learning Agents for Landmark Detection in Brain Images},
  author={Leroy, Guy and Rueckert, Daniel and Alansary, Amir},
  journal={arXiv preprint arXiv:2008.08055},
  year={2020}
}
```

```
@article{alansary2019evaluating,
  title={{Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection}},
  author={Alansary, Amir and Oktay, Ozan and Li, Yuanwei and Le Folgoc, Loic and
          Hou, Benjamin and Vaillant, Ghislain and Kamnitsas, Konstantinos and
          Vlontzos, Athanasios and Glocker, Ben and Kainz, Bernhard and Rueckert, Daniel},
  journal={Medical Image Analysis},
  year={2019},
  publisher={Elsevier}
}
 ```
