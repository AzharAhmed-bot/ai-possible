
# First training Phase

During the 1st training it had the following configuration:
- A 3 by 3 kernel matrix with 64 filters
- Max Pool matrix of 2 by 2
- 128 hidden neural networks
- 10 Epochs (training phases)

The training accuracy was **55.7%** with loss of **2.5113**. Which mean I need to increase its enhancement

## Second training Phase
During the 2nd training it had the following configuration:
- A 3 by 3 kernel matrix with 64 filters
- Max Pool matrix of 2 by 2
- 256 hidden neural networks
-10 Epochs

The training accuracy went to **56.4%** with loss of **3.4991**. Which means maybe I should increase my training phases

## Third training Phase
During the 3st training it had the following configuration:
- A 3 by 3 kernel matrix with 64 filters
- Max Pool matrix of 2 by 2
- 128 units hidden neural networks
- 10 Epochs (training phases)

The training accuracy was **53.7%** with loss of **3.5053**. Let's try increasing my convolution layers

## Fourth Training Phase
During the 4th training it had the following configuration:
- Add 2 more convolution and pooling layer
- 2 Hidden layers:
    - First with 256 units, while dropping out half of the units during training.
    - Second with 128 units, while dropping 1/3 of the units during training.

The training accuracy was **95.34%** with a loss of **0.1906**


## Fourth Training Phase
During the 4th training it had the following configuration:
- Add 1 more convolution and pooling layer
- 2 Hidden layers:
    - First with 256 units, while dropping out half of the units during training.
    - Second with 128 units, while dropping 1/3 of the units during training.
- Implemented learning rate reduction when performance plateaus
- Added BatchNormalization after each convolutional and dense layer(to cub internal covariate shift)

The training accuracy was **95.81%** with a loss of **0.1264**
