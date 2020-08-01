# RL-Medical

Deep Reinforcement Learning (DRL) agents applied to medical images

## Examples

- [Landmark detection using different DQN variants for a single agent implemented using Tensorpack](examples/LandmarkDetection/SingleAgent)
- [Landmark detection for multiple agents using different communication variants implemented in PyTorch](examples/LandmarkDetection/MultiAgent)
- [Automatic view planning using different DQN variants](examples/AutomaticViewPlanning/DQN)


## Installation

### Dependencies

tensorpack-medical requires:

+ Python=3.6
+ [tensorflow-gpu=1.14.0](https://pypi.org/project/tensorflow-gpu/)
+ [tensorpack=0.9.5](https://github.com/tensorpack/tensorpack)
+ [opencv-python](https://pypi.org/project/opencv-python/)
+ [pillow](https://pypi.org/project/Pillow/)
+ [gym](https://pypi.org/project/gym/)
+ [SimpleITK](https://pypi.org/project/SimpleITK/)

### User installation
```
pip install -U git+https://github.com/amiralansary/rl-medical.git
```

## Development

New contributors of any experience level are very welcomed

### Source code
You can clone the latest version of the source code with the command::
```
https://github.com/amiralansary/rl-medical.git
```

## Citation

If you use this code in your research, please cite these paper:

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

@inproceedings{alansary2018automatic,
  title={Automatic view planning with multi-scale deep reinforcement learning agents},
  author={Alansary, Amir and Le Folgoc, Loic and Vaillant, Ghislain and Oktay, Ozan and Li, Yuanwei and
  Bai, Wenjia and Passerat-Palmbach, Jonathan and Guerrero, Ricardo and Kamnitsas, Konstantinos and Hou, Benjamin and others},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={277--285},
  year={2018},
  organization={Springer}
}
 ```
