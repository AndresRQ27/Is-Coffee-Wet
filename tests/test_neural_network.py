import unittest

from IsCoffeeWet import neural_network

from pandas import read_csv

from IsCoffeeWet import config_file as cf

# Path for Linux
# PATH_TEST = "/media/andres/DATA/Code-Projects/Is-Coffee-Wet/resources/"
# Path for Windows
# PATH_TEST = "D:/VMWare/Shared/Is-Coffee-Wet/resources/"
# Path for Docker
PATH_TEST = "/opt/project/resources/"


class Test_TestNeuralNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sets the index using Datetime column

        cls.dataset = read_csv(PATH_TEST + "test_parsed.csv",
                               engine="c", index_col="Datetime", parse_dates=True)
        # Infers the frequency
        cls.dataset = cls.dataset.asfreq(cls.dataset.index.inferred_freq)

    def test_normalize(self):
        normalized_ds = neural_network.normalize(self.dataset)

        # Only max of those columns as the sin/cos won't change
        max_original = self.dataset[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()
        max_normalize = normalized_ds[["Temp Out", "Leaf Wet 1", "Leaf Wet Accum"]].max()

        # Max normalize must always be less than the original
        with self.subTest(msg="compare vs original"):
            # noinspection PyTypeChecker
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
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), len(datetime_index))


if __name__ == '__main__':
    unittest.main()
