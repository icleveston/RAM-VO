# RAM-VO

Paper: [RAM-VO: Less is more in Visual Odometry](https://arxiv.org/abs/2107.02974)

Thesis: [RAM-VO: A Recurrent Attentional Model for Visual Odometry](https://arxiv.org/abs/2107.02974)

## Abstract

Building vehicles capable of operating without human supervision requires the determination of the agent's pose.
Visual Odometry (VO) algorithms estimate the egomotion using only visual changes from the input images. 
The most recent VO methods implement deep-learning techniques using convolutional neural networks (CNN) extensively,
which add a substantial cost when dealing with high-resolution images. Furthermore, in VO tasks, more input data does
not mean a better prediction; on the contrary, the architecture may filter out useless information. 
Therefore, the implementation of computationally efficient and lightweight architectures is essential. 
In this work, we propose the RAM-VO, an extension of the Recurrent Attention Model (RAM) for visual odometry tasks.
RAM-VO improves the visual and temporal representation of information and implements the Proximal Policy Optimization (PPO)
to learn robust policies. The results indicate that RAM-VO can perform regressions with six degrees of freedom from monocular
input images using approximately 3 million parameters. In addition, experiments on the KITTI dataset demonstrate that 
RAM-VO achieves competitive results using only 5.7% of the available visual information.

The contributions of this work are:
- A lightweight VO method that selects the important input information via attentional mechanisms;
- The first visual odometry architecture that implements reinforcement learning in part of the pipeline;
- Several experiments on KITTI~\cite{geiger_vision_2013} sequences demonstrating the validity and efficiency of RAM-VO.

## Usage

To train a new model:
```python
python main.py
```

To test a trained model on a specific sequence:
```python
python main.py --test <out_exec_folder> --dataset 'kitti' --test_seq <sequence>
```

To generate results, such as metrics, trajectories, and plots:
```bash
./gen_results.zsh <out_exec_folder>
```
