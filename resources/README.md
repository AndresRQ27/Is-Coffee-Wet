# Relevant variables

- Temperature (Temp Out/In)
  - It's important to note that the internal temperature almost never
    fails, contrary to the outside temperature.
  - High and low temperature are also important to have in order for a
    proper analysis for the experts.
- External relative humidity (Out Hum)
- Wind speed
- Wind direction (Wind Dir)*
  - Although its a primary variable (measurable directly from a sensor), it
    won't be predicted by the model as it's a cardinal value
- Atmospheric pressure (Bar)
- Rain
  - It's important to sum all the rain that has fallen in time when
    resampling as it's important to know it for the experts.
- Solar radiation (Solar Rad)
- Soil humidity (Soil Temp)
- Leaf Wetness (Leaf Wet)

New columns added
- Leaf Wetness Accumulated (Leaf Wet Accum)
  - Total time, in minutes, the leaf has been wet during a time frame.
- "Frequency" cos/sin
  - For each frequency, adds a decomposition of sin and cos to the columns.
  - This helps capture cyclical behaviour of the weather, by letting the
    neural network analyze it by "season", like daily, yearly, by
    trimester, etc.
  - The columns created depends on the values in `encodeNames` and `encodeFreq`
    - `encodeNames` are the names fro the new columns
    - `encodeFreq` are the frequency (in seconds) of the data.