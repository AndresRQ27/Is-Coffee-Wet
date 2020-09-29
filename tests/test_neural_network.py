import unittest
from pandas import read_csv
from numpy import ones
from IsCoffeeWet import neural_network
from IsCoffeeWet import config_file as cf
from IsCoffeeWet import window_generator as wg


class Test_TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Sets the index using Datetime column
        self.dataset = read_csv("D:/VMWare/Shared/Is-Coffee-Wet/resources/test_parsed.csv",
                                engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        self.dataset = self.dataset.asfreq(self.dataset.index.inferred_freq)

        label_width = 24*7  # A week
        input_width = label_width

        # Window for testing the NN
        self.window = wg.WindowGenerator(input_width=input_width,
                                         label_width=label_width,
                                         shift=label_width,
                                         label_columns=["Temp Out",
                                                        "Leaf Wet 1",
                                                        "Leaf Wet Accum"])

        self.window._example = [ones((label_width, 3))]

    def test_normalize(self):
        normalized_ds = neural_network.normalize(self.dataset)

        # Only max of those columns as the sin/cos won't change
        max_original = self.dataset[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()
        max_normalize = normalized_ds[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max normalize must always be less than the original
        with self.subTest(msg="compare vs original"):
            self.assertTrue(all(max_original > max_normalize))

        # The normalize dataset must be less than 4 std to be correct
        # At least in this dataset
        with self.subTest(msg="check if less than 4"):
            self.assertTrue(all(4 > max_normalize))

    def test_split_dataset(self):
        config_file = cf.ConfigFile()
        config_file.training = 0.7
        config_file.validation = 0.2
        config_file.num_data = len(self.dataset)

        datetime_index, train_ds, val_ds, test_ds = neural_network.split_dataset(self.dataset, config_file)

        # Original size must remain
        self.assertEqual(len(train_ds)+len(val_ds)+len(test_ds), len(datetime_index))

    def test_convolutional_model(self):
        filter_size = 32  # Neurons in a conv layer
        conv_width = 24  # A day
        input_size = (self.window.input_width, self.dataset.shape[1])
        output_size = (self.window.label_width, len(self.window.label_columns))

        model = neural_network.convolutional_model(filter_size, conv_width, input_size, output_size)

        self.assertTrue(model.output_shape is not None)


if __name__ == '__main__':
    unittest.main()
