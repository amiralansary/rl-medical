# Anatomical Landmark Detection

Automatic detection of anatomical landmarks is an important step for a wide range of applications in medical image analysis. In this project, we formulate the landmark detection problem as a sequential decision process navigating in a medical image environment towards the target landmark. We deploy multiple Deep Q-Network (DQN) based architectures to train agents that can learn to identify the optimal path to the point of interest. This code also supports both fixed- and multi-scale search strategies with hierarchical action steps in a coarse-to-fine manner.

<p align="center">
<img style="float: center;" src="images/framework.png" width="465">
<img style="float: center;" src="images/actions.png" width="130">
</p>

---
## Results
Here are few examples of the learned agent for landmark detection on unseen data:

* Detecting the apex point in short-axis cardiac MRI [(HQ video)](videos/cardiac_apex.mp4)
<p align="center">
<img src="./images/cardiac_apex.gif" width="255">
</p>

* Detecting the anterior commissure (AC) point in adult brain MRI [(HQ video)](videos/brain_ac.mp4)
<p align="center">
<img src="./images/brain_ac.gif" width="255">
</p>

* Detecting the cavum septum pellucidum (CSP) point in fetal head ultrasound [(HQ video)](videos/fetal_csp.mp4)
<p align="center">
<img src="./images/fetal_csp.gif" width="255">
</p>


---

## Usage

### Train
```
python DQN.py --task train --algo DQN --gpu 0
```

### Test
```
python DQN.py --task play --algo DQN --gpu 0 --load data/models/DQN_multiscale_brain_mri_point_pc_ROI_45_45_45/model-600000
```

## Citation

If you use this code in your research, please cite this paper:

```
@article{alansary2018evaluating,
    title={{Evaluating Reinforcement Learning Agents for Anatomical
      Landmark Detection}},
    author={Alansary, Amir and Oktay, Ozan and Yuanwei, Li and
      Le Folgoc, Loic and Hou, Benjamin and Vaillant, Ghislain and
      Glocker, Ben and Kainz, Bernhard and Rueckert, Daniel},
    url={https://openreview.net/forum?id=SyQK4-nsz},
    year={2018}
 }
 ```
