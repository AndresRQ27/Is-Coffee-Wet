import unittest
import numpy as np
from pandas import read_csv, date_range
from IsCoffeeWet import data_parser
from IsCoffeeWet import config_file as cf


class Test_TestDataParser(unittest.TestCase):
    def setUp(self):
        self.dirtyDataset = read_csv("resources/test.csv")

    def test_merge_datetime(self):
        # Uses a list of columns with the date as the config file
        config_file = cf.ConfigFile()
        config_file.datetime = ["Date", "Time"]
        config_file.columns = ["Date", "Time"]
        config_file.datetime_format = "%d/%m/%Y %I:%M %p"
        
        datetime_ds = data_parser.merge_datetime(self.dirtyDataset, config_file)
        
        result = date_range("2010-04-07 00:00:00", 
                               "2010-04-07 00:04:00", 
                               freq="min")

        # Test to see if the resulting format is as the desired one
        # All the values must match to generate a True
        self.assertTrue((result == datetime_ds.head(5).index).all())

    def test_convert_numeric(self):
        # INFO: Needs datetime index to interpolate by time
        config_file = cf.ConfigFile()
        config_file.datetime = ["Date", "Time"]
        config_file.columns = ["Date", "Time"]
        config_file.datetime_format = "%d/%m/%Y %I:%M %p"
        dataset = data_parser.merge_datetime(self.dirtyDataset, config_file)
        
        config_file.null = ["---"]
        config_file.columns.extend(["Temp Out", "Leaf Wet 1"])
        config_file.formats = {"Leaf Wet 1": "int"}

        convert_ds = data_parser.convert_numeric(dataset, config_file)

        #? print(convert_ds.info(verbose=True))

        # Test if there is a NaN value
        with self.subTest(msg="NaN test"):
            # isna() returns False in value is NaN
            # Use all() to detect if there is a single False value (NaN)
            # First all check for True in all columns
            # Second all check for True accross all columns
            self.assertTrue((convert_ds.notna()).all().all())

        # Test if all values are converted
        with self.subTest(msg= "dtypes test"):
            # Check if all columns are float64
            self.assertTrue((convert_ds.dtypes == "float64").all())
            

    def test_sample_dataset(self):
        # INFO: Needs datetime index to resample
        config_file = cf.ConfigFile()
        config_file.datetime = ["Date", "Time"]
        config_file.columns = ["Date", "Time"]
        config_file.datetime_format = "%d/%m/%Y %I:%M %p"
        dataset = data_parser.merge_datetime(self.dirtyDataset, config_file)

        # INFO_ Needs clean dataset to resample
        config_file.null = ["---"]
        config_file.columns.extend(["Temp Out", "Leaf Wet 1"])
        config_file.formats = {"Leaf Wet 1": "int"}
        dataset = data_parser.convert_numeric(dataset, config_file)

        config_file.freq = "15min"
        config_file.functions = {"Leaf Wet 1": "last"}

        sample_ds = data_parser.sample_dataset(dataset, config_file)

        # Checks the new frequency of the dataset
        with self.subTest(msg="freq test"):
            self.assertEqual(sample_ds.index.freq, "15T")

        with self.subTest(msg="check 'Leaf Wet Accum'"):
            self.assertTrue("Leaf Wet Accum" in sample_ds.columns)



    def test_cyclical_encoder(self):
        # INFO: Needs datetime index to encode
        config_file = cf.ConfigFile()
        config_file.datetime = ["Date", "Time"]
        config_file.columns = ["Date", "Time"]
        config_file.datetime_format = "%d/%m/%Y %I:%M %p"
        dataset = data_parser.merge_datetime(self.dirtyDataset, config_file)

        config_file.encode = {"day": 86400}

        encoded_ds = data_parser.cyclical_encoder(dataset, config_file)

        #Sin/Cos columns added to the new dataset
        with self.subTest(msg="check sin/cos test"):
            self.assertTrue(("day sin" in encoded_ds.columns) 
                            and "day cos" in encoded_ds.columns)

        #Check if there values are between -1 and 1
        with self.subTest(msg="check between 0 and 1"):
            self.assertTrue((min(encoded_ds["day sin"]) >= -1)
                            and max((encoded_ds["day sin"]) <= 1))


if __name__ == '__main__':
    unittest.main()
