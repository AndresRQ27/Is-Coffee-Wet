TCN = {
    "filter_size": [160, 160, 96],
    "kernel_size": [10, 2, 2],
    "dilations": 3,
    "dropout": 0.2
}
CNN = {
    "filter_size": [96, 192, 208],
    "kernel_size": [5, 11, 24],
    "pool_size": [2, 3],
    "dropout": 0.1
}
CONV_LSTM = {
    "filter_size": [64, 64, 64, 64, 224],
    "kernel_size": [8, 4, 12, 6],
    "pool_size": 2,
    "dropout": 0.1
}