import pandas
import utils
from scipy import special
from lmfit import Model
from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import interpolate
import measurements
from grafica.plotly_utils.utils import line

LABELS_FOR_PLOTS = {
	'Normalized collected charge mean': 'Normalized collected charge',
	'Normalized collected charge median': 'Normalized collected charge',
}

def metal_silicon_transition_model_function_left_pad(x, y_scale, laser_sigma, x_offset, y_offset):
	return y_scale*special.erf((x-x_offset)/laser_sigma*2**.5) + y_offset

def metal_silicon_transition_model_function_right_pad(x, y_scale, laser_sigma, x_offset, y_offset):
	return metal_silicon_transition_model_function_left_pad(-x, y_scale, laser_sigma, -x_offset, y_offset)

def fit_erf(df, windows_size: float):
	"""Given a df with data from a single 1D scan, this function fits an
	erf (convolution of Gaussian and step) to each metal-silicon interface. 
	Returns the fit result object by lmfit, one for each pad (left and right).
	
	Parameters
	----------
	df: pandas.DataFrame
		Data frame containing the data from a 1D scan with 2 pixels.
	windows_size: float
		Size of the window that covers the two pixels. I.e. it is the
		distance from the metal→silicon interface in the left pixel to
		the silicon→metal interface in the right pixel.
	
	Returns
	-------
	fit_results: dict
		A dictionary of the form `{'left': lmfit.ModelResult, 'right': lmfit.ModelResult}`
		with the results of fitting the model to each pixel.s
	"""
	
	df = df.loc[df['n_pulse']==1] # Use only pulse 1 for this.
	df = df.loc[df['Distance (m)'].notna()] # Drop rows that have NaN values in the relevant columns.
	df = df.loc[df['Normalized collected charge'].notna()] # Drop rows that have NaN values in the relevant columns.
	
	fit_results = {}
	fit_model_left_pad = Model(metal_silicon_transition_model_function_left_pad)
	fit_model_right_pad = Model(metal_silicon_transition_model_function_right_pad)
	for pad in set(df['Pad']):
		this_pad_df = df.loc[df['Pad']==pad]
		if pad == 'left':
			indices_for_fit = this_pad_df['Distance (m)']<this_pad_df['Distance (m)'].mean()-windows_size/3
			fit_model = fit_model_left_pad
		elif pad == 'right':
			indices_for_fit = this_pad_df['Distance (m)']>this_pad_df['Distance (m)'].mean()+windows_size/3
			fit_model = fit_model_right_pad
		x_data_for_fit = this_pad_df.loc[indices_for_fit, 'Distance (m)']
		y_data_for_fit = this_pad_df.loc[indices_for_fit, 'Normalized collected charge']
		parameters = fit_model.make_params(
			laser_sigma = 10e-6,
			x_offset = this_pad_df['Distance (m)'].mean()-windows_size/2 if pad=='left' else this_pad_df['Distance (m)'].mean()+windows_size/2,
			y_scale = 1/2,
			y_offset = 1/2,
		)
		parameters['y_scale'].set(min=0, max=1)
		parameters['y_offset'].set(min=0, max=1)
		parameters['laser_sigma'].set(min=5e-6, max=22e-6)
		fit_results[pad] = fit_model.fit(y_data_for_fit, parameters, x=x_data_for_fit)
	return fit_results

def script_core(directory:Path, window_size:float, force:bool=False):
	"""Fit an ERF function to each side where the interfaces between silicon
	and metal are supposed to be (1D linear TCT scan).
	
	Parameters
	----------
	directory: Path
		Path to the main directory of a measurement.
	window_size: float
		Size of the window to shine the laser through the metalization
		of the device, in meters. This is the distance between the two
		interfaces metal→silicon (left pixel) and silicon→metal (right 
		pixel).
	force: bool, default False
		If `True` the function is applied, if `False` it is only applied
		if it was not applied before.
	"""
	Iñaqui = Bureaucrat(
		directory,
		new_measurement = False,
		variables = locals(),
	)
	measurement_handler = measurements.MeasurementHandler(Iñaqui.measurement_name)
	
	if measurement_handler.measurement_type != 'TCT 1D scan fixed voltage':
		raise ValueError(f'Measurement {repr(Iñaqui.measurement_name)} must be a "TCT 1D scan fixed voltage", but it is of type {repr(measurement_handler.measurement_type)}')
	
	if force == False and Iñaqui.job_successfully_completed_by_script('this script'):
		return
	
	with Iñaqui.verify_no_errors_context():
		if not Iñaqui.job_successfully_completed_by_script('parse_waveforms_from_scan_1D.py'):
			raise RuntimeError(f'There is no successfull run of `parse_waveforms_from_scan.py` for measurement {Iñaqui.measurement_name}, cannot proceed.')
		
		measurement_handler.tag_left_and_right_pads()
		
		if 'Normalized collected charge' not in measurement_handler.measurement_data.columns:
			normalized_charge_df = utils.calculate_normalized_collected_charge(
				measurement_handler.measurement_data,
				window_size = window_size,
				inter_pixel_distance = 100e-6,
			)
			measurement_handler.measurement_data['Normalized collected charge'] = normalized_charge_df['Normalized collected charge']
		
		data_df = measurement_handler.measurement_data # Shorthand notation...
		
		fit_results = fit_erf(data_df, windows_size=window_size)
		results = pandas.DataFrame(columns = ['Pad'])
		results.set_index('Pad', inplace=True)
		for pad in fit_results:
			results.loc[pad,'Laser sigma (m)'] = fit_results[pad].params['laser_sigma'].value
			results.loc[pad,'Metal-silicon distance (m)'] = fit_results[pad].params['x_offset'].value
			results.loc[pad,'y_offset'] = fit_results[pad].params['y_offset'].value
			results.loc[pad,'y_scale'] = fit_results[pad].params['y_scale'].value
		results.to_csv(Iñaqui.processed_data_dir_path/Path('fit_results.csv'))
		
		fig = line(
			data_frame = utils.mean_std(data_df, by=['n_position','Pad', 'Distance (m)']),
			x = 'Distance (m)',
			y = 'Normalized collected charge median',
			error_y = 'Normalized collected charge MAD_std',
			error_y_mode = 'band',
			color = 'Pad',
			markers = '.',
			title = f'ERF fit to silicon-metal interfaces<br><sup>Measurement: {Iñaqui.measurement_name}</sup>',
			labels = LABELS_FOR_PLOTS,
		)
		for pad in results.index:
			if pad == 'left':
				df = data_df.loc[data_df['Distance (m)']<data_df['Distance (m)'].mean()-window_size/4,'Distance (m)']
			else:
				df = data_df.loc[data_df['Distance (m)']>data_df['Distance (m)'].mean()+window_size/4,'Distance (m)']
			x = np.linspace(min(df), max(df), 99)
			fig.add_trace(
				go.Scatter(
					x = x,
					y = fit_results[pad].eval(params=fit_results[pad].params, x = x),
					mode = 'lines',
					name = f'Fit erf {pad} pad, σ<sub>laser</sub>={fit_results[pad].params["laser_sigma"].value*1e6:.1f} µm',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.write_html(str(Iñaqui.processed_data_dir_path/Path(f'erf_fit.html')), include_plotlyjs = 'cdn')
		
		# Now center data in Distance (m) = 0 and find calibration factor ---
		x_50_percent = {}
		for pad in results.index:
			if pad == 'left':
				df = data_df.loc[data_df['Distance (m)']<data_df['Distance (m)'].mean()-window_size/4,'Distance (m)']
			else:
				df = data_df.loc[data_df['Distance (m)']>data_df['Distance (m)'].mean()+window_size/4,'Distance (m)']
			x = np.linspace(min(df), max(df), 99)
			y = fit_results[pad].eval(params=fit_results[pad].params, x = x)
			inverted_erf = interpolate.interp1d(
				x = y,
				y = x,
			)
			x_50_percent[pad] = float(inverted_erf(.5))
		multiply_distance_by_this_scale_factor_to_fix_calibration = window_size/((x_50_percent['left']-x_50_percent['right'])**2)**.5
		offset_before_scale_factor_multiplication = ((x_50_percent['left']+x_50_percent['right'])**2)**.5/2
		with open(Iñaqui.processed_data_dir_path/Path('distance_calibration.txt'), 'w') as ofile:
			print(f'multiply_distance_by_this_scale_factor_to_fix_calibration = {multiply_distance_by_this_scale_factor_to_fix_calibration}', file=ofile)
			print(f'offset_before_scale_factor_multiplication = {offset_before_scale_factor_multiplication}', file=ofile)
		
		data_df['Distance from center (m)'] = data_df['Distance (m)'] - offset_before_scale_factor_multiplication
		for distance_col in {'Distance (m)','Distance from center (m)'}:
			data_df[f'{distance_col} calibrated'] = data_df[distance_col]*multiply_distance_by_this_scale_factor_to_fix_calibration
		fig = line(
			data_frame = utils.mean_std(data_df, by=['n_position','Pad', 'Distance (m) calibrated', 'Distance from center (m) calibrated']),
			x = f'Distance from center (m) calibrated',
			y = 'Normalized collected charge median',
			error_y = 'Normalized collected charge MAD_std',
			error_y_mode = 'band',
			color = 'Pad',
			markers = '.',
			title = f'Scan after applying the calibration factor (x←{multiply_distance_by_this_scale_factor_to_fix_calibration:.2e}×(x-{offset_before_scale_factor_multiplication:.2e}))<br><sup>Measurement: {Iñaqui.measurement_name}</sup>',
			labels = LABELS_FOR_PLOTS,
		)
		for pad in results.index:
			if pad == 'left':
				df = data_df.loc[data_df['Distance (m)']<data_df['Distance (m)'].mean()-window_size/4,f'Distance from center (m) calibrated']
			else:
				df = data_df.loc[data_df['Distance (m)']>data_df['Distance (m)'].mean()+window_size/4,f'Distance from center (m) calibrated']
			x = np.linspace(min(df), max(df), 99)
			fig.add_trace(
				go.Scatter(
					x = x*multiply_distance_by_this_scale_factor_to_fix_calibration,
					y = fit_results[pad].eval(params=fit_results[pad].params, x = x+offset_before_scale_factor_multiplication),
					mode = 'lines',
					name = f'Fit erf {pad} pad',
					line = dict(color='black', dash='dash'),
				)
			)
		fig.write_html(str(Iñaqui.processed_data_dir_path/Path(f'after offset subtraction and calibration.html')), include_plotlyjs = 'cdn')

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser(description='Fits an ERF function to the metal-silicon interfaces of a linear scan and uses this information to calculate a calibration factor based on the real distance.')
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement. If "all", the script is applied to all linear scans.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(
		Path(args.directory), 
		window_size = 215e-6, # From the microscope pictures.
		force = True,
	)
