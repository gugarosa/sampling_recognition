# Neighbor-based Bag of Samplings for Person Identification Through Handwritten Dynamics and Convolutional Neural Networks

*This repository holds all the necessary code to run the very-same experiments described in the paper "Neighbor-based Bag of Samplings for Person Identification Through Handwritten Dynamics and Convolutional Neural Networks".*

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

## Structure

  * `data/`: The data folder itself;
  * `datasets/`
    * `sampled.py`: A Sampled class that helps a user dealing with the bag-of-samplings approach over the raw signals;
  * `models/`
    * `alexnet.py`: An AlexNet architecture implementation;
    * `cifar10.py`: A CifarNet architecture implementation;
    * `lenet.py`: A LeNet architecture implementation;
  * `utils/`
    * `loader.py`: Methods that assist in loading the raw signals;
    * `sampler.py`: Methods that assist in sampling the raw signals.

---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

---

### Downloading the Data

One can [download](http://recogna.tech/files/person_handPD.tar.gz) the already pre-processed NewHandPD dataset. After downloading the file, decompress it and move it to the `data/` folder. Thus, its path should look like this: `data/handpd/`.

Each `handpd/` folder will contain an integer identifier (person identifier) and twelve `.txt` signals, as follows:

* (1, 2): Circles;
* (3, 4): Diakinesis;
* (5, 6, 7, 8): Meanders;
* (9, 10, 11, 12): Spirals.

---

## Usage

### Training an Architecture

After gathering the data, now it is possible to train an architecture. There are currently two ways in order to fulfill that purpose:

* `sampled_split.py`: Samples the data and trains a network using a train/test split approach;
* `sampled_kfolds.py`: Samples the data and trains a network using k-fold cross-validation.

### Calculating Metrics

Each one of the training scripts will output a `.pkl` file, which contains valuable information about the training process. One can run the following script to calculate some metrics from that process:

```Python
python calculate_metrics.py
```

### Plotting Graphics

Finally, one can use one of the following scripts in order to generate a graphic regarding the architectures' comparison:

* `plot_computational_load.py`: Plots the computational load (training time) of the architectures;
* `plot_loss_convergence.py`: Plots the training loss convergence of the architectures.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
