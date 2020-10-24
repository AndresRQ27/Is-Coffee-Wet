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

`docker run -it --name coffee-debug -v $(pwd):/data is-coffee-wet:cpu bash`

#### With GPU

- Build the image:

`docker build -f Dockerfile.gpu -t is-coffee-wet:gpu .`

- Run the container:

`docker run -it --name coffee-gpu-debug -v $(pwd):/data -gpus all is-coffee-wet:gpu bash`

### For production

This following instructions does:

- Creates a container from the production image
- Opens a terminal using bash
- Maps a volume name _neural-network_ to `/data/neural-network`
- Binds the resources folder where the host console is running to `/data/resources`
- Name the container _coffee-production_

`docker run -it 
-v neural-network:/data/neural-network 
-v $PWD/resources:/data/resources 
--name coffee-production 
is-coffee-wet:release bash`