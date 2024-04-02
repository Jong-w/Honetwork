
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
```
python MDM_main.py \
    --env_name='FrostbiteNoFrameskip-v4' \
    --gamma_5=0.999 \
    --gamma_4=0.999 \
    --gamma_3=0.999 \
    --gamma_2=0.999 \
    --gamma_1=0.999 \
    --hidden-dim-Hierarchies = [16, 256, 256, 256, 256]\
    --time_horizon_Hierarchies = [1, 10, 15, 20, 25]
```



## Result
Results for the performance of MDM on the atari environments are depicted here.



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