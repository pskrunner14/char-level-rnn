# char-level-RNN

Character Level RNN language model in PyTorch.

## Getting Started

```
$ python train.py --help
Usage: train.py [OPTIONS]

  Trains a character-level Recurrent Neural Network in PyTorch.

  Args: optional arguments [python train.py --help]

Options:
  -f, --filename PATH          path for the training data file [data/names]
  -rt, --rnn-type TEXT         type of RNN layer to use [LSTM]
  -nl, --num-layers INTEGER    number of layers in RNN [2]
  -dr, --dropout FLOAT         dropout value for RNN layers [0.5]
  -es, --emb-size INTEGER      size of the each embedding [64]
  -hs, --hidden-size INTEGER   number of hidden RNN units [256]
  -n, --num-epochs INTEGER     number of epochs for training [50]
  -bz, --batch-size INTEGER    number of samples per mini-batch [32]
  -lr, --learning-rate FLOAT   learning rate for the adam optimizer [0.0002]
  -se, --save-every INTEGER    epoch interval for saving the model [10]
  -ns, --num-samples INTEGER   number of samples to generate after epoch
                               interval [5]
  -sp, --seed-phrase TEXT      seed phrase to feed the RNN for sampling
                               [SOS_TOKEN]
  -sa, --sample-every INTEGER  epoch interval for sampling new sequences [5]
  --help                       Show this message and exit.
```

## Built With

* Python
* PyTorch
* NumPy