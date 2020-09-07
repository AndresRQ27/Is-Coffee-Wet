_En el documento plan de trabajo se explica el proceso a seguir. Vamos a tener los datos de 29 estaciones, pero por ahora solo trabajaremos con una. 
Luego aplicamos la todos nuestros modelos a las 29 para ver los resultados._

# Is-Coffee-Wet

## Before you start

Set up your `$PYTHONPATH` to include this directory (`Is-Coffee-Wet`) as
the file hierarchy needs it.

### In Ubuntu
1. Open a terminal in your Home folder
2. Open `~/.bashrc` in your favorite text editor (e.g. nano)
3. Go to the last line and write `export PYTHONPATH=$PYTHONPATH:/path/to/folder/Is-Coffee-Wet`
4. Save & exit, then close the terminal
5. Re-open the terminal and type `echo $PYTHONPATH` to check if it worked

## Requirements

The project was run using:

- Python 3.8.2 (at least >3.4)
- Pandas 1.1.1
- Numpy 1.18.5
- Matplotlib 3.3.1
- Keras 2.4.3
- Tensorflow 2.4.0 (nightly)

Optionals (GPU enhancement):

- Cuda 11.0
- CuDNN-11 v8.0
