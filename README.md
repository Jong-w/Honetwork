
## From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals


This code is for "From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals". Our work introduced hierarchy-drop, novel approach that dynamically tailors hierarchies in accordance with environmental needs to improve the efficiency and effectiveness of learning in complex environments.

Authors: Jong Won Kim, Dongjae Kim

## Model Architecture
![이미지](./img/fig_1.png)

## Installation

To run the code, please follow these steps:
1. Clone the repository.
First, clone the repository to your local machine using Git. Open your terminal, and enter the following command:
```
git clone https://github.com/Jong-w/MDM.git
```
Install dependencies using the requirements.txt file:

2. Navigate to the Project Directory
Change into the cloned repository's directory:
```shell
cd MDM
```

3. Install the dependencies
Install the required Python packages defined in the requirements.txt file:
```
pip install -r requirements.txt
```


## Train

To train our model, execute the following command. This example uses the FrostbiteNoFrameskip-v4 environment, but the parameters can be adjusted to fit other environments or specific requirements. The gamma values represent the discount factors for different levels of the hierarchy, affecting how future rewards are valued at each level. The hidden-dim-Hierarchies and time_horizon_Hierarchies parameters define the size of the hidden layers and the time horizons for the goals at each hierarchical level, respectively. Adjusting these parameters allows for fine-tuning the model's performance and learning efficiency.

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
We demonstrate the efficacy of our approach through extensive experiments. Our model shows significant improvements over baseline models in various environments.


<p align="center">
<img src="./img/FROSTBITE.png" width="300" height="200">
<img src="./img/Gravitar.png" width="300" height="200">
<img src="./img/Qbert.png" width="300" height="200">
<img src="./img/Kangaroo.png" width="300" height="200">
</p>


## Citation
If you find our work useful in your research, please consider citing:
```
@article{EfftoEff,
  title={From Extensive to Efficient: Dynamic Hierarchical Reinforcement Learning withAdaptive Long Short-Term Goals},
  author={Jong Won Kim, Dongjae Kim},
  journal={~arxiv~},
  year={2024}
}
```