from pathlib import Path
import pandas
import warnings
import numpy as np
from scipy.stats import median_abs_deviation

k_MAD_TO_STD = 1.4826 # https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation

def tag_left_right_pad(data_df):
	"""Given a data_df with data from a single 1D scan of two pads of a 
	device, this function creates a new column indicating if the pad is 
	"left" or "right" and returns such column as a data frame with the
	same index as `data_df`.
	"""
	channels = set(data_df['n_channel'])
	if len(channels) != 2:
		raise ValueError(f'`data_df` contains data concerning more than two channels. I can only tag left and right pads for two channels data.')
	left_data = data_df.loc[(data_df['n_position']<data_df['n_position'].mean())]
	right_data = data_df.loc[(data_df['n_position']>data_df['n_position'].mean())]
	for channel in channels:
		if left_data.loc[left_data['n_channel']==channel, 'Collected charge (V s)'].mean(skipna=True) > left_data.loc[~(left_data['n_channel']==channel), 'Collected charge (V s)'].mean(skipna=True):
			mapping = {channel: 'left', list(channels-{channel})[0]: 'right'}
		else:
			mapping = {channel: 'right', list(channels-{channel})[0]: 'left'}
	pad_df = pandas.DataFrame(index=data_df.index)
	for n_channel in set(data_df['n_channel']):
		pad_df.loc[data_df['n_channel']==n_channel, 'Pad'] = mapping[n_channel]
	return pad_df

def calculate_normalized_collected_charge(df:pandas.DataFrame, window_size:float, inter_pixel_distance:float, laser_sigma:float=9e-6):
	"""Calculates the normalized collected charge between 0 and 1 using 
	the mean values, not just the `max` and `min`.
	
	Parameters
	----------
	df: pandas.DataFrame
		Data frame with the data of a TCT 1D scan.
	window_size: float
		Size of the window in the metalization where the laser is shined
		through. This size is for the whole window from one pixel to the
		other, not half of the window.
	laser_sigma: float, default 9e-6.
		Approximate size of the laser.
	inter_pixel_distance: float, default 100e-6
		Approximate value of the inter-pixel distance in order to roughly
		know where to expect signal.
	
	Returns
	-------
	Return a single-column-dataframe containing the value of the 
	normalized collected charge at each row with the same index as `df`.
	"""
	normalized_charge_df = pandas.DataFrame(index=df.index)
	normalized_charge_df['Normalized collected charge'] = df['Collected charge (V s)'].copy()
	for n_pulse in sorted(set(df['n_pulse'])):
		for pad in {'left','right'}:
			rows_where_I_expect_no_signal_i_e_where_there_is_metal = (df['Distance (m)'] < df['Distance (m)'].median() - window_size/2 - 2*laser_sigma) | (df['Distance (m)'] > df['Distance (m)'].median() + window_size/2 + 2*laser_sigma)
			if pad == 'left':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] > df['Distance (m)'].median() - window_size/2 + 2*laser_sigma) & (df['Distance (m)'] < df['Distance (m)'].median() - inter_pixel_distance/2 - 2*laser_sigma)
			elif pad == 'right':
				rows_where_I_expect_full_signal_i_e_where_there_is_silicon = (df['Distance (m)'] < df['Distance (m)'].median() + window_size/2 - 2*laser_sigma) & (df['Distance (m)'] > df['Distance (m)'].median() + inter_pixel_distance/2 + 2*laser_sigma)
			offset_to_subtract = normalized_charge_df.loc[rows_where_I_expect_no_signal_i_e_where_there_is_metal&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] -= offset_to_subtract
			scale_factor = normalized_charge_df.loc[rows_where_I_expect_full_signal_i_e_where_there_is_silicon&(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'].median()
			normalized_charge_df.loc[(df['Pad']==pad)&(df['n_pulse']==n_pulse),'Normalized collected charge'] /= scale_factor
	return normalized_charge_df

def mean_std(df, by):
	"""Groups by `by` (list of columns), calculates mean and std, and creates one column with mean and another with std for each column not present in `by`.
	Example
	-------
	df = pandas.DataFrame(
		{
			'n': [1,1,1,1,2,2,2,3,3,3,4,4],
			'x': [0,0,0,0,1,1,1,2,2,2,3,3],
			'y': [1,2,1,1,2,3,3,3,4,3,4,5],
		}
	)

	mean_df = utils.mean_std(df, by=['n','x'])
	
	produces:
	
	   n  x    y mean     y std
	0  1  0  1.250000  0.500000
	1  2  1  2.666667  0.577350
	2  3  2  3.333333  0.577350
	3  4  3  4.500000  0.707107
	"""
	def MAD_std(x):
		return median_abs_deviation(x, nan_policy='omit')*k_MAD_TO_STD
	with warnings.catch_warnings(): # There is a deprecation warning that will be converted into an error in future versions of Pandas. When that happens, I will solve this.
		warnings.simplefilter("ignore")
		mean_df = df.groupby(by=by).agg(['mean','std',np.median,MAD_std])
	mean_df.columns = [' '.join(col).strip() for col in mean_df.columns.values]
	return mean_df.reset_index()
