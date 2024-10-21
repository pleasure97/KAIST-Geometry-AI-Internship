<div align=center>
  <h1>
    KAIST Geometry AI Lab Internship
  </h1>
</div>

<details open>
<summary><b>Table of Content</b></summary>
<br>

- <b> [PointNet] </b>
  - [X] Task 1 : Point Cloud Classfication
  - [X] Task 2 : Point Cloud Segmentation
  - [X] Task 3 : Point Cloud Auto-Encoding

- <b> [NeRF] </b>
  - [X] Task 1 : Implementing MLP 
  - [X] Task 2 : Implementing Ray Sampling 
  - [X] Task 3 : Volume Rendering Equation 
  - [X] Task 4 : Positional Encoding 
  - [X] Task 5 : Evaluations 
  - [X] Task 6 : NeRF with Custom Data

- <b> [NeRF-Studio] </b>
  - [X] Task 1 : NeRFStudio - Run NeRFactory
  - [X] Task 2 : NeRFStudio - Try with Custom Data 
  - [X] Task 3 : NeRFStudio - Run Other Methods 
  - [X] Task 4 : SDFStudio - Run NeuS
  - [X] Task 5 : SDFStudio - Export Results 
  - [X] Task 6 : ThreeStudio - Run DreamFusion 
  - [X] Task 7 : ThreeStudio - Run Other Methods

- <b> [Diffusion Model] </b>
  - [X] Task 1 : Training Diffusion Model 
  - [X] Task 2 : Sampling Diffusion Model 
  - [X] Task 3 : Evaluation 
  - [X] Task 4 : DDPM 
  - [X] Task 5 : Classifier-Free Guidance 
  - [X] Task 6 : Image Inpainting 

- <b> [Stable Diffusion] </b>
  - [X] Task 1 : Introduction to HuggingFace and Diffusers 
  - [X] Task 2 : A Simple Practice on Stable Diffusion & ControlNet 
  - [X] Task 3 : Extending Stable Diffusion to Arbitrary Resolutions 

- <b> [Research Project] </b>
  - [X] Motivation 
  - [X] Problem Definition 
  - [X] Method
  - [X] Intermediate Results 
  - [X] Limitations

</details>


## Code Structure
```
.
├── PointNet
│   ├── checkpoints               <--- 
│   ├── data                      <--- 
│   ├── dataloaders               <--- 
│   ├── utils                     <--- 
│   ├── model.py                  <--- 
│   ├── practice.py               <--- 
│   ├── train_ae.py               <--- 
│   ├── train_cls.py              <--- 
│   └── train_seg.py              <--- 
├── NeRF
│   ├── media                     <--- 
│   ├── scripts                   <--- 
│   ├── environment.yaml          <--- 
│   ├── requirements.txt          <--- 
│   └──  torch_nerf               <--- 
│       ├── configs               <--- 
│       ├── runners               <--- 
│       └── src                   <--- 
│              ├── cameras               <--- 
│              ├── network               <--- 
│              ├── renderer              <--- 
│              ├── scene                 <--- 
│              ├── signal_encoder        <--- 
│              └── utils                 <--- 
├── Diffusion
│   ├── image_diffusion           <--- 
│       ├── dataset.py            <--- 
│       ├── ddpm.py               <--- 
│       ├── module.py             <--- 
│       ├── network.py            <--- 
│       ├── sampling.py           <--- 
│       ├── scheduler.py          <--- 
│       ├── train.py              <--- 
│       └── fid                   <--- 
│   └── SDE
│       ├── dataset.py            <--- 
│       ├── eval.py               <--- 
│       ├── loss.py               <--- 
│       ├── network.py            <--- 
│       ├── sampling.py           <--- 
│       ├── sde.py                <--- 
│       └── train.py              <--- 
└── Assets
```

## PointNet


## NeRF


## NeRF-Studio

## Diffusion Model 

## Stable Diffusion 

## Research Project 



## Resources
- [[paper](https://arxiv.org/abs/2011.13456)] Score-Based Generative Modeling through Stochastic Differential Equations
- [[paper](https://arxiv.org/abs/2006.09011)] Improved Techniques for Training Score-Based Generative Models
- [[paper](https://arxiv.org/abs/2006.11239)] Denoising Diffusion Probabilistic Models
- [[paper](https://arxiv.org/abs/2105.05233)] Diffusion Models Beat GANs on Image Synthesis
- [[paper](https://arxiv.org/abs/2207.12598)] Classifier-Free Diffusion Guidance
- [[paper](https://arxiv.org/abs/2010.02502)] Denoising Diffusion Implicit Models
- [[paper](https://arxiv.org/abs/2206.00364)] Elucidating the Design Space of Diffusion-Based Generative Models
- [[paper](https://arxiv.org/abs/2106.02808)] A Variational Perspective on Diffusion-Based Generative Models and Score Matching
- [[paper](https://arxiv.org/abs/2305.16261)] Trans-Dimensional Generative Modeling via Jump Diffusion Models
- [[paper](https://openreview.net/pdf?id=nioAdKCEdXB)] Likelihood Training of Schrödinger Bridge using Forward-Backward SDEs Theory
- [[blog](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)] What is Diffusion Model?
- [[blog](https://yang-song.net/blog/2021/score/)] Generative Modeling by Estimating Gradients of the Data Distribution
- [[lecture](https://youtube.com/playlist?list=PLCf12vHS8ONRpLNVGYBa_UbqWB_SeLsY2)] Charlie's Playlist on Diffusion Processes
- [[slide](./assets/summary_of_DDPM_and_DDIM.pdf)] Juil's presentation slide of DDIM
- [[slide](./assets/sb_likelihood_training.pdf)] Charlie's presentation of Schrödinger Bridge.
