# MusicGenerator

## Overview

Havard CS182 Fall 2018 AI Final Project

## Usage

Put training data inside `/data/midi/ragtimemusic`. After modifying the training data, make sure to delete everything inside `/data/samples`.

To train the model, simply run `main.py`. Once trained, you can generate the results with `main.py --test --sample_length 500`. For more help and options, use `python main.py -h`.

The saved music will be in `/save/model/midi`.

To visualize the computational graph and the cost with TensorBoard, run `tensorboard --logdir save/`.


## Credits

Credits for the original code go to [Conchylicultor](https://github.com/Conchylicultor/MusicGenerator). 
