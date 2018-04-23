# Anatomical Landmark Detection

Standard view images are important in clinical practice as they provide a means
to perform biometric measurements from similar anatomical regions.
In this project, we employ a multi-scale reinforcement learning (RL) agent
framework that enables a natural learning paradigm by interacting with the
environment and mimicking experienced operators' navigation steps.

<p align="center">
<img style="float: center;" src="images/framework.png" width="512">
</p>


## Usage

### Train
```
python DQN.py --algo DQN --gpu 0
```

### Test
```
python DQN.py --algo DQN --gpu 0 --task play --load path_to_trained_model
```

## Citation

If you use this code in your research, please cite this paper:

```
@unpublished{alansary2018automatic,
  title={{Automatic View Planning with Multi-scale Deep Reinforcement Learning Agents}},
  author={Alansary, Amir and Le Folgoc, Loic and Vaillant, Ghislain and Oktay, Ozan and Li, Yuanwei and Bai, Wenjia and Passerat-Palmbach, Jonathan and Guerrero, Ricardo and Kamnitsas, Konstantinos and Hou, Benjamin and McDonagh, Steven and Glocker, Ben and Kainz, Bernhard and Rueckert, Daniel},
  year={2018},
  note={Under review}
  }
 ```
