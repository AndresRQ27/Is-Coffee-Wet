#! Code obtain from https://www.tensorflow.org/tutorials/structured_data/time_series#1_indexes_and_offsets

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ! Dataset must be at least of size 300
# but be serious, that size is not even enough to learn anything

class WindowGenerator:
    """
    Window used for training that uses consecutive samples from the data to
    make a set of predictions. Can adjust the number of inputs to receive,
    the predictions to make and how far in the future will they be; for one
    particular feature or different features at the same time.
    """

    def __init__(self, input_width, label_width, shift,
                 train_ds=None, val_ds=None, test_ds=None,
                 label_columns=None, batch_size=64):
        """
        Initialization class. Assigns the datasets to the object and creates
        the necessary objects (slices, enums) for the neural network to go
        over during the training and prediction.

        Can be started without the dataset but no use for training the
        neural network; useful if you want to measure the capabilities of
        the window with the parameters used.

        Parameters
        ----------
        input_width: int
            Number of timestamps to take as input.
        label_width: int
            Size of the prediction to be made.
        shift: int
            Amount of time to shift the prediction.
        train_ds, val_ds, test_ds: pandas.DataFrame, optional
            Dataset used for training, validation and testing. Stored as a
            DataFrame and converted to `tensorflow.data.Dataset` when its
            property is called.
        label_columns: list[string], optional
            Names of the columns to make predictions. Columns not included
            here will be used for the input but not the output. By default
            it's empty, so no label would be predicted.
        batch_size: int, optional
            Size of the batch when feeding the dataset to the NN. By
            default, the batch size is 32.

        Notes
        -----
        If the `label_width` is the same size as `shift`, the prediction
        made will be the same as the time shifted (e.g. if shifted 7 days
        into the future, the prediction will be for those 7 days).

        If the `label_width` is less than `shift`, a non-consecutive
        prediction into the future will be returned (e.g. if shifted 7 days
        but label is 1 day, all the days would be predicted internally but
        only the 7th day would be return as the prediction).
        """

        # Store the raw data (whatever type it is, but generally DataFrame).
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        # Work out the label column indices.
        # Associate each label column with a number to use as internal reference
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}

        # Do the same with the input columns for the network
        # Remember that not all columns in the input are predicted (only labels)
        if train_ds is not None:
            self.column_indices = {name: i for i, name in
                                   enumerate(train_ds.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        # Size of the batches when creating the tensorflow.data.Dataset
        self.batch_size = batch_size

        # Calculates the total amount of time-steps the window will take
        self.total_window_size = input_width + shift

        # Slices used to travel the dataset
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        # The label (prediction) will always start counting from the end
        self.label_start = self.total_window_size - self.label_width

        # Slices used to travel the dataset
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        """
        Function called when printing the object. Returns the shape of the
        window and label columns that will predict.

        Returns
        -------
        string
            String with the information of the window and labels.
        """

        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        """
        Function that will convert a list of consecutive inputs into a
        window of inputs (for the NN) and a window of labels. It also
        handles the `label_columns` for single and multiple outputs.

        Parameters
        ----------
        features: tensorflow.data.Dataset
            Input data of the function.

        Returns
        -------
        tuple[tensorflow.data.Dataset, tensorflow.data.Dataset]
            Pair of `tensorflow.data.Dataset` divided into the input and
            the label.
        """

        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, plot_col, model=None, max_subplots=3):
        """
        Function that plots an example taken from the train dataset to show
        the accuracy of the predictions vs the real values.

        Parameters
        ----------
        plot_col: string
            Column name of the data to plot.
        model: tensorflow.keras.Model, optional
            Model to obtain the predictions when graphing. Only works if
            `plot_col` is in `label_columns`
        max_subplots: int, optional
            Maximum amount of subplots to show.
        """
        # Takes the inputs and labels from the example
        inputs, labels = self.example

        # Sets the figure size
        plt.figure(figsize=(12, 8))

        # Gets the number of the index to plot from the input dictionary
        plot_col_index = self.column_indices[plot_col]

        # Resolves the amount of plots to do
        max_n = min(max_subplots, len(inputs))

        # Plots the points in the graph
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            # Gets the number of the index to plot from the label dictionary
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)

            # If column isn't a label, then no labels are show in the graph
            # Use the plot_col as only the input
            else:
                label_col_index = plot_col_index

            # Don't graph the labels and continue with the next iteration
            if label_col_index is None:
                continue

            # Graphs the label points
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            # Generates the predictions to graph
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        """
        Function that converts the a `pandas.DataFrame` into a
        `tensorflow.data.Dataset` to use in a neural network model.

        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with the information of the dataset
        Returns
        -------
        tensorflow.data.Dataset
            Dataset transform into a `tensorflow.data.Dataset`
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_ds)

    @property
    def val(self):
        return self.make_dataset(self.val_ds)

    @property
    def test(self):
        return self.make_dataset(self.test_ds)

    @property
    def example(self):
        """
        Get and cache an example batch of `inputs, labels` for plotting
        from the training set.
        """
        result = getattr(self, '_example', None)

        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result

        return result
