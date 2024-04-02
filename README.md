
## From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals


This code is for "From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals".

Authors: Jong Won Kim, Dongjae Kim

![이미지](./img/fig_1.png)

## Usage

Clone this repository to use MDM.
1. Clone the repository.
```
git clone https://github.com/Jong-w/MDM.git
```
Install dependencies using the requirements.txt file:

2. Move directory
```shell
cd MDM
```

3. Install the dependencies
```
pip install -r requirements.txt
```
## Train

python MDM_main.py \
    --outdir=output_path \
    --gpus=8 \
    ~~~(요런 느낌의 하이퍼파라미터)

## Inference




## Result
Results for the performance of RMADDPG and QMIX on the Particle Envs and QMIX in SMAC are depicted here. 
These results are obtained using a normal (not prioitized) replay buffer. (요거 수정)

(환경 나눈 것 설명)
<p align="center">
<img src="./img/Env_classification.png" width="400" height="220">
</p>

(결과 plot)



<p align="center">
<img src="./img/FROSTBITE.png" width="300" height="200">
<img src="./img/Gravitar.png" width="300" height="200">
<img src="./img/Qbert.png" width="300" height="200">
<img src="./img/Kangaroo.png" width="300" height="200">
</p>


## Citation
```
@article{EfftoEff,
  title={From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals},
  author={Jong Won Kim, Dongjae Kim},
  journal={~~~},
  year={2024}
}
```