![Demonstration of a ConvCNP](https://github.com/cambridge-mlg/convcnp/blob/master/demo_images/convcnp.gif)

# Convolutional Conditional Neural Processes

This repository contains code for the 1-dimensional experiments from
[Convolutional Convolutional Neural Processes](https://openreview.net/forum?id=Skey4eBYPS). The code for the 2-dimensional experiments can be found [here](https://yanndubs.github.io/Neural-Process-Family/reproducibility/ConvCNP.html).

* [Installation](#installation)
* [Expository Notebooks](#expository-notebooks)
* [Reproducing the 1D Experiments](#reproducing-the-1d-experiments)
* [Reference](#reference)

## Installation
Requirements:

* Python 3.6 or higher.

* `gcc` and `gfortran`:
    On OS X, these are both installed with `brew install gcc`.
    On Linux, `gcc` is most likely already available,
    and `gfortran` can be installed with `apt-get install gfortran`.
    
To begin with, clone and enter the repo.

```bash
git clone https://github.com/cambridge-mlg/convcnp
cd convcnp
```

Then make a virtual environment and install the requirements.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This will install the latest version of `torch`.
If your version of CUDA is not the latest version, then you might need to
install an earlier version of `torch`.

You should now be ready to go!
If you encounter any problems, feel free to open an issue, and will try to
help you resolve the problem as soon as possible.

Common issues:

* `fatal error: Python.h: No such file or directory`:
    Python libraries seem to be missing.
    Try `sudo apt-get install python3.X-dev` with `X` replaced by your
    particular version.

## Expository Notebooks
For a tutorial-style exposition of ConvCNPs, see the following two
expository notebooks:

* [Implementing and Training Convolutional Conditional Neural Processes](https://github.com/cambridge-mlg/convcnp/blob/master/convcnp_regression.ipynb), and
* [Sequential Inference with Convolutional Conditional Neural Processes](https://github.com/cambridge-mlg/convcnp/blob/master/sequential_inference.ipynb).

## Reproducing the 1D Experiments
To reproduce the numbers from the 1d experiments,
`python train.py <data> <model> --train` can be used.
The first argument, `<data>`, specifies the data that the model will be trained
on, and should be one of the following:
 
* `eq`: samples from a GP with an exponentiated quadratic (EQ) kernel;
* `matern`: samples from a GP with a Matern-5/2 kernel;
* `noisy-mixture`: samples from a GP with a mixture of two EQ kernels and
    some noise;
* `weakly-periodic`: samples from a GP with a weakly-periodic kernel; or
* `sawtooth`: random sawtooth functions.

The second argument, `<model>`, specifies the model that will be trained,
and should be one of the following:

* `convcnp`: small architecture for the Convolutional Conditional Neural
    Process;
* `convcnpxl`: large architecture for the Convolutional Conditional Neural
    Process;
* `cnp`: Conditional Neural Process; or
* `anp`: Attentive Conditional Neural Process.

Upon calling `python train.py <data> <model> --train`, first the specified
model will be trained on the specified data source. Afterwards, the script
will print the average log-likelihood on unseen data.

To reproduce the numbers from all the 1d experiments from the paper at once, you
can use `./run_all.sh`.

For more options, please see `python train.py --help`:

```
usage: train.py [-h] [--root ROOT] [--train] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
                {eq,matern,noisy-mixture,weakly-periodic,sawtooth}
                {convcnp,convcnpxl,cnp,anp}

positional arguments:
  {eq,matern,noisy-mixture,weakly-periodic,sawtooth}
                        Data set to train the CNP on.
  {convcnp,convcnpxl,cnp,anp}
                        Choice of model.

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           Experiment root, which is the directory from which the
                        experiment will run. If it is not given, a directory
                        will be automatically created.
  --train               Perform training. If this is not specified, the model
                        will be attempted to be loaded from the experiment
                        root.
  --epochs EPOCHS       Number of epochs to train for.
  --learning_rate LEARNING_RATE
                        Learning rate.
  --weight_decay WEIGHT_DECAY
                        Weight decay.
```


## Reference

Gordon, J., Bruinsma W. P., Foong, A. Y. K., Requeima, J., Dubois Y.,
Turner, R. E.
(2019).
"Convolutional Conditional Neural Processes,"
_International Conference on Learning Representations (ICLR), 8th_.

 BiBTeX:

```
@inproceedings{Gordon:2020:Convolutional_Conditional_Neural_Processes,
    title = {Convolutional Conditional Neural Processes},
    author = {Jonathan Gordon and Wessel P. Bruinsma and Andrew Y. K. Foong and James Requeima and Yann Dubois and Richard E. Turner},
    year = {2020},
    booktitle = {International Conference on Learning Representations},
    url = {https://openreview.net/forum?id=Skey4eBYPS}
}
```
