# Using Docker

## Requirements

- Docker v19.03

### Optionals (GPU enhancement):

- NVIDIA Drivers
- NVIDIA Container Toolkit

An installation guide is available
[here](https://github.com/NVIDIA/nvidia-docker)

## To run the container

`docker container run --name is-coffee-wet -v $(pwd):/opt/project/resources`

# Using Linux or Windows

## Requirements

The project was run using:

- Numpy 1.16
- Matplotlib 3.3
- Python 3.6 (at least >3.4)
- Pandas 1.1
- Tensorflow 2.3
- Tensorflow Addon 0.11

### Optionals (GPU enhancement):

- Cuda 11.0
- Cudnn-11 v8.0

## Before you start

Set up your `$PYTHONPATH` to include this directory (`Is-Coffee-Wet`) as
the file hierarchy needs it.

### In Ubuntu
1. Open a terminal in your Home folder
2. Open `~/.bashrc` in your favorite text editor (e.g. nano)
3. Go to the last line and write `export PYTHONPATH=$PYTHONPATH:/path/to/folder/Is-Coffee-Wet`
4. Save & exit, then close the terminal
5. Re-open the terminal and type `echo $PYTHONPATH` to check if it worked
