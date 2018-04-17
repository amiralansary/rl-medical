# tensorpack-medical

Extension to [tensorpack](https://github.com/ppwwyyxx/tensorpack) for medical
imaging applications.

## Examples
Here are some selected examples of medical imaging with deep learning research.
In these examples we try to replicate the results from recent published work,
so that it can be used for actual research comparisons.

+ Reinforcement Learning:
  - [Landmark detection using different DQN variants](examples/LandmarkDetection/DQN)
  - [Automatic view planning using different DQN variants](examples/AutomaticViewPlanning/DQN)
+ Supervised Learning:
  - [Image segmentation](examples) [todo]
  - [Saliency maps](examples) [todo]


## Installation

### Dependencies

tensorpack-medical requires:

+ Python >= 3.5
+ TensorFlow >= 1.6.0
+ Tensorpack >= 0.8.0
+ Python OpenCV
+ pyglet

### User installation
```
pip install -U git+https://github.com/amiralansary/tensorpack-medical.git
```

## Development

New contributors of any experience level are very welcomed. Reproducing more
examples from recent published works is very helpful for the reproducibility of
new research.

### Source code
You can clone the latest version of the source code with the command::
```
git clone https://github.com/amiralansary/tensorpack-medical.git
```
