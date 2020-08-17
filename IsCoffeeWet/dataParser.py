import pandas as pd
import numpy as np
import warnings


def cleanDataset(dataset, removeRows=[], nullValue="", removeColumns=[]):
    """
    Removes missing or empty values that can generate error in the parse 
    of the dataset

    Returns a dataset without "damaged" values that could hurt a convertion
    of a `dtype` in a column of the dataset

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to clean.
    - removeRows: list or tuple of strings.
        Columns where to search the null values.
        Wrong name will result in a failed removal.
        Default value is an empty list.
    - nullValue:  object, generally string or int.
        Value to search and remove.
        Default value is an empty string.
    - removeColumns: list or tuple of strings.
        Group of columns to remove. Wrong names will
        result in a failed removal. Default value is 
        an empty list.

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with entire columns remove and/or
        rows with incomplete data removed.
    """
    print()  # ! Print
    # Removes the columns with almost all missing values
    displayNumber = 0  # Iterator to print
    if removeColumns:  # Checks if empty
        print("Column removal started...")  # ! Print
        for columnName in removeColumns:  # Remove
            try:
                dataset = dataset.drop(columnName, axis=1)
                displayNumber += 1
            except (KeyError):
                np.mean()
                print("Column doesn't exist:", columnName)

    print("Columns removed:", displayNumber)

    # Sets al the missing values (represented as nullValue) to NaN in the
    # selected columns
    displayNumber = dataset.size  # Iterator to print
    if removeRows:
        print("Rows removal started...")  # ! Print
        for rowName in removeRows:
            try:
                dataset = dataset.replace({rowName: nullValue}, np.NaN)
            except (KeyError):
                print("Column doesn't exist:", columnName)

    # Removes the NaN values in the rows
    dataset = dataset.dropna(axis=0, how="any")

    print("Rows removed:", displayNumber-dataset.size)

    return dataset


def setDataTypes(dataset, mergeDateTime=[], nameFormat=[]):
    """
    Reorders the dataset by merging Date & Time into a 
    pd.datetime64[ns]. It also sets the types of the 
    rest of the columns according to the `dtype` passed.

    Returns a dataset with the column casted to the desired
    values for easier manipulation

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to change the `dtype`.
    - mergeDateTime: tuple of strings.
        A two strings tuple that has in the first position
        the name of the column of the dates. In the second 
        position, the name of the time. Extra parameters will
        be ignored and one parameter will result in a failed
        conversion. By default, its an empty list
    - nameFormat: list of tuples.
        Each tuple is made of 2 parameters: 
        the name (string) of the column and the `dtype` in a string 
        to be converted. Wrong name or incorrect casting will result
        in a failed conversion. By default is an empty list. Example: 
        >>> [("Temp Out", "float32")]

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with entire columns remove and/or
        rows with incomplete data removed.
    """
    print()  # ! Print
    if mergeDateTime:
        try:
            print("Date and Time merge started...")  # ! Print
            # Unifies "Date" and "Time" in a "datetime64[ns]"
            dataset["Datetime"] = pd.to_datetime(dataset[mergeDateTime[0]] +
                                                 ' ' + dataset[mergeDateTime[1]])
            # Sets the datetime as the index of the DataFrame
            dataset = dataset.set_index('Datetime')
            # Drops the "Time" columns as it has been combine into "Date"
            dataset = dataset.drop(
                [mergeDateTime[0], mergeDateTime[1]], axis=1)
        except (KeyError):
            print("Incorrect column name. Check",
                  mergeDateTime[0], mergeDateTime[1])
        except (IndexError):
            print("Incorrect number of parameters in mergeDateTime")
        except (ValueError):
            print("Column can't be change to Datetime")

    print("Date and Time merge finished...")  # ! Print

    # Changes the data to types that use less memory
    if nameFormat:
        print("Values conversion started...")  # ! Print
        for nameAndType in nameFormat:
            try:
                dataset = dataset.astype({nameAndType[0]: nameAndType[1]})
            except (KeyError):
                print("Incorrect column name:", nameAndType[0])
            except (ValueError):
                print(str(nameAndType[0]) +
                      "can't be change to" + str(nameAndType[1]))

    print("Values conversion finished...")  # ! Print

    return dataset


def sampleDataset(dataset, columnFunction=[], frequency="15min"):
    """
    Sample dataset according to a time frequency given. The dataset 
    will be group in interval of `frequency` and a numeric function will
    be applied to the rows between these timeframes to get one single value
    for each new timestamp.

    Different numeric functions can be applied to different column by 
    pairing the function with the column in a tuple in `columnFunction`.
    If a column is in the dataset but not in `columnFunction` or doesn't
    have a function assigned for the sampling, it will be ignore (and 
    removed) in the returned dataset.

    Parameters
    ----------
    - dataset: pd.DataFrame.
        Dataset to sample each column.
    - columnFunction: list of tuples.
        Each tuple is made of 2 parameters: 
        the name (string) of the column and the function to apply in
        the sample. Wrong name or incorrect casting will result
        in a failed conversion. By default is an empty list. Example: 
        >>> [("Temp Out", "np.mean)]
    - frequency: string. Value expressed in string of the frequency to
        sample the data. By default, the dataset gets sample in groups
        of 15min. Other examples are:
        >>> "1H" --An hour
        >>> "1D" --A day
        >>> "1M" --A month
        >>> "1h30min" --Combination of hours and minutes

    Returns
    -------
    - dataset : pd.DataFrame.
        Dataset with the rows sample in the desired frequency.
        If no sample could be made, return the original dataset and
        a warning will be given.
    """
    # Creates the new dataset to return
    newDataset = pd.DataFrame()

    print("Resampling started...")  # ! Print

    for filterFunction in columnFunction:
        try:
            newDataset = pd.concat([newDataset,  # Concatenates to the right the result in the new dataset
                                    dataset.resample(frequency).agg(  # Samples a specific column with its corresponding function
                                        {filterFunction[0]:
                                         filterFunction[1]})],
                                   axis=1)
        except (KeyError):
            print("Incorrect column name:", filterFunction[0])
        except (pandas.core.base.DataError):
            print("Incorrect function to apply to:", filterFunction[0])

    print("Resampling finished...")  # ! Print
    print("Clearing NA values in resampled dataset...")  # ! Print

    # Cleans the dataset from extra values generated while sampling
    newDataset = newDataset.dropna(axis=0, how="any")

    # If it's empty, return the original dataset
    if newDataset.empty:
        warnings.warn("Returning un-sampled dataset", Warning)  # ! Warning
        newDataset = dataset

    return newDataset
