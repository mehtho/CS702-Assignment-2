"""
This scripts pads the missing data in the original data. The missing data is filled with NaNs. 

.csv data format:
t_ms,x_px,y_px
2000,1650,657
2100,1636,665
2200,1621,668
2300,1609,668
2400,1595,673
...
6000, NaN, NaN (padded missing data)
...

The time step is 100ms, starting from 2000ms. Note that the missing part is between 6000 - 9000. 
"""
import pandas as pd

data = pd.read_csv("./q3/txys_missingdata.csv")
start_time, end_time = data['t_ms'].min(), data['t_ms'].max()
full_range = pd.DataFrame({'t_ms': range(start_time, end_time + 100, 100)})
padded_data = pd.merge(full_range, data, on='t_ms', how='left')
padded_data.to_csv('./q3/txys_missingdata_padded.csv', index=False)