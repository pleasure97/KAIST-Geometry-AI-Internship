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
│   ├── checkpoints                      <--- Checkpoints for Point Cloud Models 
│   ├── data                             <--- Store the dataset when learning ShapeNet 
│   ├── dataloaders                      <--- Load the dataset when learning ShapeNet 
│   ├── utils                            <--- Utils for Pre & Post Processing 
│   ├── model.py                         <--- Codes for Related Neural Network Models 
│   ├── practice.py                      <--- Briefly Check the performance of the model 
│   ├── train_ae.py                      <--- Point Cloud Auto Encoding 
│   ├── train_cls.py                     <--- Point Cloud Classification 
│   └── train_seg.py                     <--- Point Cloud Segmentation 
├── NeRF
│   ├── media                            <--- Store the NeRF's rendering output 
│   ├── scripts                          <--- Script files for downloading the dataset and creating videos
│   ├── environment.yaml                  
│   ├── requirements.txt                  
│   └──  torch_nerf                     
│       ├── configs                      <--- Configurations for Inital Settings such as CUDA, NNs, and Renderer
│       ├── runners                      <--- Codes for Training, Evaluating, and Rendering 
│       └── src                          
│              ├── cameras               <--- Codes for Camera Class and RaySampling Class 
│              ├── network               <--- Code for NeRF base class
│              ├── renderer              <--- Codes for Volumetric Rendering 
│              ├── scene                 <--- Codes for creating the simple primitive mesh 
│              ├── signal_encoder        <--- Codes for NeRF's positional encoding 
│              └── utils                 <--- Codes for Loading datasets and Evaluation Metrics 
├── Diffusion
│   ├── image_diffusion                  
│       ├── dataset.py                   <--- Codes for loading and processing AFHQ dataset 
│       ├── ddpm.py                      <--- Codes for DDPM(Denoising Diffusion Probablistic Models)
│       ├── module.py                    <--- Codes for Basic Network Modules such as UpSampling, ResBlock... 
│       ├── network.py                   <--- Code for UNet 
│       ├── sampling.py                  <--- Codes for creating the sample with the learned model 
│       ├── scheduler.py                 <--- Codes for learning rate scheduler 
│       ├── train.py                     <--- Codes for training a DDPM or DDIM based model
│       └── fid                          <--- Codes for an evaluation metric called Frechet Inception Distance
│   └── SDE
│       ├── dataset.py                   <--- Codes for extracting samples according to the type of the dataset 
│       ├── eval.py                      <--- Codes for calculating the similarity of point clouds 
│       ├── loss.py                      <--- Codes for losses such as Schrodinger Bridge Loss
│       ├── network.py                   <--- Codes for positional encoding, MLP, and Simple Network 
│       ├── sampling.py                  <--- Codes for Sampler 
│       ├── sde.py                       <--- Codes for Variance Preserving SDE, Variance Exploding SDE 
│       └── train.py                     <--- Codes for showing the training results in matplotlib  
└── Assets
```

---

**## PointNet**

---

**## NeRF**

---

**## NeRF-Studio**

---

**## Diffusion Model**

---

**## Stable Diffusion**

---

**## Research Project**

---