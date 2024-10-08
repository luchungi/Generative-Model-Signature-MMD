# Code for "Generative model for financial time series trained with MMD using a signature kernel"

## Lu Chung I, Julian Sester

# Abstract

Generating synthetic financial time series data that accurately reflects real-world market dynamics holds tremendous potential for various applications, including portfolio optimization, risk management, and large scale machine learning. We present an approach for training generative models for financial time series using the maximum mean discrepancy (MMD) with a signature kernel. Our method leverages the expressive power of the signature transform to capture the complex dependencies and temporal structures inherent in financial data. We employ a moving average model to model the variance of the noise input, enhancing the model's ability to reproduce stylized facts such as volatility clustering. Through empirical experiments on S&P 500 index data, we demonstrate that our model effectively captures key characteristics of financial time series and outperforms a comparable GAN-based approach. In addition, we explore the application of the synthetic data generated to train a reinforcement learning agent for portfolio management, achieving promising results. Finally, we propose a method to add robustness to the generative model by tweaking the noise input so that the generated sequences can be adjusted to different market environments with minimal data.

# Preprint

[Link to paper](https://arxiv.org/abs/2407.19848)

# Contents

The notebooks are used for training:
1. The [generative model](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/train_model.ipynb) using the MMD with a signature kernel
2. The [reinforcement learning agent](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/trading.ipynb) using the trained generative model
3. The [robust reinforcement learning agent](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/robust_trading.ipynb) using the trained generative model with robust noise

The pytorch model architecture is in ['generators.py'](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/model/generators.py).  
The signature kernel related functions are in [kernels.py](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/kernels.py) which is based on code modified from this [repo](https://github.com/tgcsaba/KSig).  
The loss function to train the generative model is in ['loss.py'](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/sigkernel/loss.py).  
All data related functions can be found in ['data.py'](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/utils/data.py).  
The reinforcement learning environment for portfolio management is in [env.py](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/utils/env.py).  
The Lambert transformation function is in [gaussianize.py](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/utils/gaussianize.py) which is taken from this [repo](https://github.com/gregversteeg/gaussianize).  
All training related routines are in ['train.py](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/train.py).  

The S&P 500 data sourced from Yahoo Finance in [/data](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/data/).  
The figures omitted from the paper can be found in [/figures](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/figures/).  
Training related metrics can be found in the tensorboard logs in [/runs](https://github.com/luchungi/Generative-Model-Signature-MMD/blob/main/runs/).  
