from pathlib import Path
import pandas
from bureaucrat.Bureaucrat import Bureaucrat
import utils
from scipy.stats import median_abs_deviation
import numpy as np

MEASUREMENTS_DATA_PATH = Path('../measurements_data')

def measurement_type(measurement_name: str) -> str:
	"""Returns the type of measurement (beta scan, TCT scan, IV, etc.).
	
	Example
	-------
	```
	MEASUREMENTS = {
		'20220404021350_MS07_1DScan_228V',
		'20220404021350_MS07_1DScan_228V',
		'20220403214116_MS07_sweeping_bias_voltage',
		'20220328170454_MS03_IV_Curve',
		'20220405203845_MS08_3_probe_station',
		'20220317155531_BetaScan_SpeedyGonzalez12_at_98V',
	}
	for m in MEASUREMENTS:
		print(f'{m} | {measurement_type(m)}')
	```
	produces
	```
	20220328170454_MS03_IV_Curve | IV curve
	20220404021350_MS07_1DScan_228V | TCT 1D scan fixed voltage
	20220317155531_BetaScan_SpeedyGonzalez12_at_98V | beta fixed voltage
	20220405203845_MS08_3_probe_station | IV curve probe station
	20220403214116_MS07_sweeping_bias_voltage | TCT 1D scan sweeping bias voltage
	```
	"""
	measurement_type = 'unknown'
	sub_directories_in_this_measurement = [p.parts[-1] for p in (MEASUREMENTS_DATA_PATH/Path(measurement_name)).iterdir()]
	if 'beta' in measurement_name.lower():
		measurement_type = 'beta'
		if 'sweeping' not in measurement_name.lower():
			measurement_type += ' fixed voltage'
		elif 'sweeping_bias_voltage' in measurement_name.lower():
			measurement_type += ' voltage scan'
	elif 'iv' in measurement_name.lower() and 'curve' in measurement_name.lower():
		measurement_type = 'IV curve'
	if all([s in measurement_name.lower() for s in {'probe','station'}]):
		measurement_type = 'IV curve probe station'
	elif '1DScan' in measurement_name and 'scan_1D' in sub_directories_in_this_measurement:
		measurement_type = 'TCT 1D scan fixed voltage'
	elif 'sweeping_bias_voltage' in measurement_name.lower() and 'scan_1D_sweeping_bias_voltage' in sub_directories_in_this_measurement:
		measurement_type = 'TCT 1D scan sweeping bias voltage'
	return measurement_type

def read_and_preprocess_measurement_TCT_1DScan_fixed_voltage(measurement_name: str):
	"""Reads the "main data" from a TCT measurement and does some pre-processing
	so it is easier and more standard in the top level scripts to handle
	it.
	
	Returns
	-------
	data_df: pandas.DataFrame
		Data frame with the data.
	"""
	data_df = pandas.read_feather(MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('parse_waveforms_from_scan_1D')/Path('data.fd'))
	return data_df

def read_and_preprocess_IV_curve_done_at_the_beta_setup(measurement_name: str):
	"""Reads the data from an IV curve made in the beta setup. This will
	fail for the probe station.
	"""
	return pandas.read_feather(MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('iv_curve')/Path('measured_data.fd'))

def read_and_preprocess_IV_curve_done_at_the_TCT_setup(measurement_name: str):
	"""Reads the data from an IV curve made in the TCT setup. This will
	fail for the probe station.
	"""
	df = pandas.read_feather(MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('IV_curve')/Path('measured_data.fd'))
	for col in {'Bias current (A)','Bias voltage (V)'}:
		df[col] = df[col].abs()
	return df

def read_and_preprocess_IV_done_at_the_probe_station(measurement_name: str):
	return pandas.read_feather(MEASUREMENTS_DATA_PATH/Path(measurement_name)/Path('convert_probe_station_measurement_to_our_format')/Path('measured_data.fd'))

class MeasurementHandler:
	"""A class specialized in handling the measurements of the ETL
	Market Survey.
	"""
	def __init__(self, measurement_name: str):
		"""Creates a `MeasurementHantler`.
		
		Parameters
		----------
		measurement_name: str
			Name (not path) of a measurement.
		"""
		if not isinstance(measurement_name, str):
			raise TypeError(f'`measurement_name` must be an instance of {type("hola")}, received object of type {type(measurement_name)}.')
		self._measurement_name = measurement_name
	
	@property
	def measurement_name(self) -> str:
		return self._measurement_name
	
	@property
	def measurement_type(self) -> str:
		"""Returns the measurement type, e.g. "TCT 1D scan sweeping bias voltage"
		or "IV curve probe station". To see all possible options refer
		to the source code.
		"""
		if not hasattr(self, '_measurement_type'):
			self._measurement_type = measurement_type(self.measurement_name)
		return self._measurement_type
	
	@property
	def measurement_data(self) -> pandas.DataFrame:
		"""Reads the "main data" from this measurement and does some pre-processing
		so it is easier and more standard in the top level scripts to handle
		it.
		
		Returns
		-------
		data_df: pandas.DataFrame
			Data frame with the data.
		"""
		if not hasattr(self, '_measurement_data_df'):
			if self.measurement_type == 'TCT 1D scan fixed voltage':
				self._measurement_data_df = read_and_preprocess_measurement_TCT_1DScan_fixed_voltage(self.measurement_name)
			elif self.measurement_type == 'IV curve':
				try:
					self._measurement_data_df = read_and_preprocess_IV_curve_done_at_the_beta_setup(self.measurement_name)
				except FileNotFoundError:
					self._measurement_data_df = read_and_preprocess_IV_curve_done_at_the_TCT_setup(self.measurement_name)
			elif self.measurement_type == 'IV curve probe station':
				self._measurement_data_df = read_and_preprocess_IV_done_at_the_probe_station(self.measurement_name)
			else:
				raise NotImplementedError(f"Don't know how to read a measurement of type {repr(self.measurement_type)}.")
		return self._measurement_data_df
	
	def tag_left_and_right_pads(self) -> None:
		"""If the measurement is one in which we can define which is the
		left and right pads (e.g. a TCT 1D scan), it adds a column to the
		measured data with such tag. Otherwise, a `TypeError` is rised.
		"""
		if self.measurement_type == 'TCT 1D scan fixed voltage':
			if 'Pad' not in self.measurement_data.columns:
				pads_df = utils.tag_left_right_pad(self.measurement_data)
				self.measurement_data['Pad'] = pads_df['Pad']
		else:
			raise TypeError(f"Don't know how to tag left and right pads for a measurement of type {repr(self.measurement_type)}.")
	
	@property
	def bias_voltage_summary(self):
		"""Returns information related to the bias voltage. Because we 
		are dealing with several different types of measurements (IV curves,
		beta scans, TCT scans, etc.) the returned object will vary. It may
		return a string e.g. "no information", it may return the list of
		bias voltages in the case of an IV curve, etc. You have to check
		when you call this method what is it giving to you.
		"""
		if not hasattr(self, '_bias_voltage'):
			if self.measurement_type == 'TCT 1D scan fixed voltage':
				self._bias_voltage = {
					'mean': self.measurement_data['Bias voltage (V)'].mean(),
					'median': self.measurement_data['Bias voltage (V)'].median(),
					'std': self.measurement_data['Bias voltage (V)'].std(),
					'MAD_std': median_abs_deviation(self.measurement_data['Bias voltage (V)'], nan_policy='omit')*utils.k_MAD_TO_STD,
				}
			else:
				raise NotImplementedError(f'Dont know how to summarize for measurement of type {repr(self.measurement_type)}.')
		return self._bias_voltage
	
	@property
	def inter_pixel_distance_summary(self):
		"""Tries to return the inter pixel distance, in case it is possible.
		If the handler is handling a measurement for which it has no
		sense to speak of inter-pixel distance, a `NotImplementedError` is raised.
		The returned object depends on the type of measurement (e.g. a
		TCT scan at a fixed voltage or a collection of TCT scans at different
		bias voltages).
		"""
		if not hasattr(self, '_inter_pixel_distance_summary'):
			if self.measurement_type == 'TCT 1D scan fixed voltage':
				try:
					with open(MEASUREMENTS_DATA_PATH/Path(self.measurement_name)/Path('calculate_inter_pixel_distance_for_single_1D_scan/interpixel_distance.txt'),'r') as ifile:
						for line in ifile:
							if 'Inter-pixel distance (m) = ' in line:
								inter_pixel_distance = float(line.split(' = ')[-1])
					with open(MEASUREMENTS_DATA_PATH/Path(self.measurement_name)/Path('calculate_inter_pixel_distance_for_single_1D_scan/interpixel_distance_bootstrapped_values.txt'),'r') as ifile:
						bootstrapped_replicas = []
						for line in ifile:
							bootstrapped_replicas.append(float(line))
				except (FileNotFoundError, ValueError):
					pass
				_locals_now = locals()
				if any([has_to_be not in _locals_now for has_to_be in {'inter_pixel_distance','bootstrapped_replicas'}]) or not (MEASUREMENTS_DATA_PATH/Path(self.measurement_name)/Path('calculate_inter_pixel_distance_for_single_1D_scan/.script_successfully_applied')).is_file():
					raise RuntimeError(f'No information (or no reliable information) about the inter-pixel distance could be found for measurement {self.measurement_name}.')
				self._inter_pixel_distance_summary = {
					'Inter-pixel distance (m) value on data': inter_pixel_distance,
					'Inter-pixel distance (m) mean': np.mean(bootstrapped_replicas),
					'Inter-pixel distance (m) median': np.median(bootstrapped_replicas),
					'Inter-pixel distance (m) std': np.std(bootstrapped_replicas),
					'Inter-pixel distance (m) MAD_std': median_abs_deviation(bootstrapped_replicas)*utils.k_MAD_TO_STD,
				}
			else:
				raise NotImplementedError(f'Dont know how to summarize for measurement of type {repr(self.measurement_type)}.')
		return self._inter_pixel_distance_summary
	
	@property
	def distance_calibration(self):
		"""Returns the distance calibration factor produced by the "fit
		ERF" script if it makes sense for this type of measurement and
		if it was calculated beforehand. Otherwise, rises error.
		"""
		if not hasattr(self, '_distance_calibration'):
			if self.measurement_type == 'TCT 1D scan fixed voltage':
				try:
					with open(MEASUREMENTS_DATA_PATH/Path(self.measurement_name)/Path('fit_erf_and_calculate_calibration_factor/distance_calibration.txt'),'r') as ifile:
						for line in ifile:
							if 'multiply_distance_by_this_scale_factor_to_fix_calibration = ' in line:
								distance_calibration_factor = float(line.split(' = ')[-1])
							if 'offset_before_scale_factor_multiplication = ' in line:
								offset_factor = float(line.split(' = ')[-1])
				except (FileNotFoundError, ValueError):
					pass
				_locals_now = locals()
				if any([must_be_in_locals not in _locals_now for must_be_in_locals in {'distance_calibration_factor','offset_factor'}]) or not (MEASUREMENTS_DATA_PATH/Path(self.measurement_name)/Path('fit_erf_and_calculate_calibration_factor/.script_successfully_applied')).is_file():
					raise RuntimeError(f'No information (or no reliable information) about the distance calibration factor could be found for measurement {self.measurement_name}.')
				self._distance_calibration = {
					'scale factor': distance_calibration_factor, 
					'offset before scale': offset_factor
				}
			else:
				raise NotImplementedError(f'Dont know how to get a distance calibration factor for measurement of type {repr(self.measurement_type)}.')	
		return self._distance_calibration
	
	@property
	def measured_devices(self) -> list:
		"""If possible, returns a list with the names of the devices that
		were measured. Otherwise raises `RuntimeError`.
		"""
		if not hasattr(self, '_measured_devices'):
			measured_devices = []
			for s in self.measurement_name.split('_'):
				if s[:2] == 'MS':
					measured_devices.append(s)
			if len(measured_devices) == 0:
				raise RuntimeError(f'Cannot find the measured devices for meausrement {self.measurement_name}.')
			self._measured_devices = measured_devices
		return self._measured_devices
	
if __name__ == '__main__':
	MEASUREMENTS = {
		'20220404021350_MS07_1DScan_228V',
		'20220404021350_MS07_1DScan_228V',
		# ~ '20220403214116_MS07_sweeping_bias_voltage',
		# ~ '20220328170454_MS03_IV_Curve',
		# ~ '20220405203845_MS08_3_probe_station',
		# ~ '20220317155531_BetaScan_SpeedyGonzalez12_at_98V',
	}
	for m in MEASUREMENTS:
		handler = MeasurementHandler(m)
		print(handler.measurement_name)
		print(handler.measurement_type)
		handler.tag_left_and_right_pads()
		print(handler.measurement_data)
