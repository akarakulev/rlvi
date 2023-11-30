# Robust Learning via Variational Inference (RLVI)

This code implements Adaptive Parameter-Free Robust Learning using Latent Bernoulli Variables.

## Install dependencies
```
pip3 install -r requirements.txt
```

## Run the experiments

### Standard parameter estimation
Benchmark, reproduced from [(Osama et al., 2020)](https://doi.org/10.1109/OJSP.2020.3039632), that compares robust learning algorithms on the four test problems with outliers in the training data: linear and logistic regression, principal component analysis, covariance estimation.
```
cd standard-learning
python3 main.py
```

### Online learning
Binary classification for Human Activity Recognition dataset [(Amine El Helou, 2023)](https://www.mathworks.com/matlabcentral/fileexchange/54138-sensor-har-recognition-app), performed in batches with varying noise to simulate the online learning setting.
```
cd online-learning
python3 main.py
```

### Deep learning
#### Synthetic noise (MNIST, CIFAR10, CIFAR100)
Experiments with the datasets in which training data is corrupted with synthetic noise.
There are three types of noise: `symmetric`, `asymmetric`, and `pairflip`. Noise rate from 0 to 1 needs to be specified.
For the method, one can use `rlvi` or one of the following: `regular`, `coteaching` [(Han et al., 2018)](https://papers.nips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html), `jocor` [(Wei et al., 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.pdf), `cdr` [(Xia et al., 2020)](https://openreview.net/forum?id=Eql5b1_hTE4), and `usdnl` [(Xu et al., 2023)](https://doi.org/10.1609/aaai.v37i9.26264).

Example:
```
cd deep-learning
python3 main.py \
        --method=rlvi \
        --dataset=mnist \
        --noise_type=pairflip \
        --noise_rate=0.45
```

#### Real noise (Food101)
Experiments with the dataset in which training data is corrupted by nature: some of the training images are mislabeled and contain some noise.
For the method, one can specify `rlvi` or one of the following: `regular`, `coteaching`, `jocor`, `cdr`, and `usdnl`.

Example:
```
cd deep-learning
python3 food.py --method=rlvi
```