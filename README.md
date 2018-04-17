# tensorpack-medical

Extension to [tensorpack](https://github.com/ppwwyyxx/tensorpack) for medical imaging applications.

## Examples
Here are some selected examples of medical imaging with deep learning research. In
these examples we try to replicate the results from recent published work, so that
it can be used for actual research comparisons.

+ Reinforcement Learning:
  - [Landmark detection using different DQN variants](examples/LandmarkDetection/DQN)
  - [Automatic view planning using different DQN variants](examples/AutomaticViewPlanning/DQN)
+ Supervised Learning:
  - [Image segmentation](examples) [todo]
  - [Saliency maps](examples) [todo]


## Installation

Dependencies:
+ Python >= 3.5
+ TensorFlow >= 1.6.0
+ Python OpenCV
+ pyglet

```
pip install -U git+https://github.com/amiralansary/tensorpack-medical.git
```
