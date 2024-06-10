# Some cool experiments with diffusion

With this repo you will be able to train small scale diffusion models based on [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) and Denoising Diffusion Probabilistic Models
## Example Diffusion Model on MINST dataset with contextual embedding
### generating `0`
![Generated images](/data/nice.png)
### generating `3`
![Generated images](/data/nice2.png)

### Bigger scale/color images
Even tho I work most of the time with minst dataset I tried to train in on bigger dataset. But this kind of training requires a bit more time and bigger model
Samole images from `900k` param model (Subset of Imagenet dataset, 64x64px)
![ImageNet](main/data/imagenet.png)

### sampling 
![Anim of generating MINST digit](main/data/Animation.gif)

# How to reproduce?
## Training
You can just install dependencies (`python 3.9`, `tensorflow=2.12.1`) (on `WSL`) and run one of the training scprit with default args
```
~/pyrepos/diffusion  main ✔                                                                         10m  ⍉
› python experiments/train_2.py
```
* `train_1.py` - Will train on minst dataset without embedding
* `train_2.py` - Will train model on minst with embedding
* `train_3.py` - Will train model on subset of ImageNet with embedding

Depending on task, dataset and net size you will need to wait from few epoch few hundred. MINST is pritty easy to train and will take few minutes with modern GPU.

## Generating
Generating scripts can be used to generate images with given model and arguments. check `exp_2_batch.py` to generate images of MINST dataset
```
~/pyrepos/diffusion  main ✔                                                                                                                                                                 9m  
› python exp_2_batch.py --model '/home/lord225/pyrepos/diffusion/retain/model-2-41.weights.h5' --c 5
```
Output:
![Output of command](/data/reproduce_out.png)
