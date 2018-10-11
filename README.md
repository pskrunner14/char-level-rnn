# char-level-RNN

Character Level RNN language model in PyTorch.

## Getting Started

```
$ python train.py --help
Usage: train.py [OPTIONS]

  Trains a character-level Recurrent Neural Network in PyTorch.

  Args: optional arguments [python train.py --help]

Options:
  -f, --filename PATH          path for the training data file
  -es, --emb-size INTEGER      size of the each embedding
  -hs, --hidden-size INTEGER   number of hidden RNN units
  -n, --num-epochs INTEGER     number of epochs for training
  -bz, --batch-size INTEGER    number of samples per mini-batch
  -lr, --learning-rate FLOAT   learning rate for the adam optimizer
  -se, --save-every INTEGER    epoch interval for saving the model
  -sa, --sample-every INTEGER  epoch interval for sampling new sequences
  --help                       Show this message and exit.
```

## Built With

* Python
* PyTorch
* NumPy