# Using Docker

## Requirements

- Docker v19.03

### Optionals (GPU enhancement):

- NVIDIA Drivers
- NVIDIA Container Toolkit

An installation guide is available
[here](https://github.com/NVIDIA/nvidia-docker)

## To run the container

### Debug

#### With CPU

- Build the image:

`docker build -f Dockerfile.cpu -t is-coffee-wet:cpu .`

- Run the container:

`docker run -it --name coffee-debug -v $(pwd):/workspaces is-coffee-wet:cpu bash`

#### With GPU

- Build the image:

`docker build -f Dockerfile.gpu -t is-coffee-wet:gpu .`

- Run the container:

`docker run -it --name coffee-gpu-debug -v $(pwd):/workspaces -gpus all is-coffee-wet:gpu bash`

### For production

This following instructions does:

- Creates a container from the production image
- Opens a terminal using bash
- Maps a volume name _checkpoints_ to `/workspaces/checkpoints`
- Binds the resources folder where the host console is running to `/workspaces/resources`
- Name the container _coffee-production_

`docker run -it 
-v checkpoints:/workspaces/checkpoints 
-v $PWD/resources:/workspaces/resources 
--name coffee-production 
is-coffee-wet:release bash`

# Running the program

Run the following command from `/workspaces`:

`python ./IsCoffeeWet/main.py $PATH_TO_CONFIG-FILE`

The path to the config file is from the resources folder. For example, if
the config file is in `/workspaces/resources/config/config.json`, the path to the
config file should be `config/config.json`.