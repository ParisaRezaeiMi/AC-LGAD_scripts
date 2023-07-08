from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from grafica.plotly_utils.utils import set_my_template_as_default
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import utils

class RSDReconstructor:
	def fit(self, positions, features):
		"""Fit the model using the training data.
		
		Arguments
		---------
		positions: array like
			An `n_events`×`2` array containing as columns the positions
			in x and y.
		something: array like
			An `n_events`×`n_features` array with e.g. the amplitude.
		"""
		for k,i in {'positions':positions,'features':features}.items():
			if not isinstance(i, pandas.DataFrame):
				raise TypeError(f'`{k}` must be an instance of pandas data frame, but received object of type {type(i)}. ')
		if len(positions.shape) != 2 or positions.shape[1] != 2:
			raise ValueError(f'`positions` is not a `n_events×2` array. I expect to receive two columns (with the values of x and y) and n_events rows, but received an array of shape {positions.shape}. ')
		if len(features.shape) != 2 or len(features)!=len(positions):
			raise ValueError(f'`features` does not have the expected dimensionality and/or length.')
		
		self.scalers = {_:MinMaxScaler(copy=False) for _ in (list(positions.columns)+list(features.columns))}
		for xy in positions.columns: # This iterates over something like `['x (m)','y (m)']`.
			self.scalers[xy].fit(positions[[xy]])
		for feature in features.columns:
			self.scalers[feature].fit(features[[feature]])
		
		self._positions_columns = positions.columns
		self._features_columns = features.columns
	
	def reconstruct(self, features):
		"""Perform a reconstruction of positions given the observed features.
		
		Arguments
		---------
		something: array like
			An `n_events_to_reconstruct`×`n_features` array with the 
			features in order to perform a reconstruction.
			
		Returns
		-------
		reconstructed_positions: array like
			An `n_events_to_reconstruct`×2 array with the reconstructed
			positions.
		"""
		for k,i in {'features':features}.items():
			if not isinstance(i, pandas.DataFrame):
				raise TypeError(f'`{k}` must be an instance of pandas data frame, but received object of type {type(i)}. ')
		if set(features.columns) != set(self._features_columns):
			raise ValueError(f'`features` columns do not match the ones used during the fitting. During the fitting process I received {sorted(set(self._features_columns))} and now for reconstruction I am receiving {sorted(set(features.columns))}. ')

class SVMReconstructor(RSDReconstructor):
	def fit(self, positions, features):
		super().fit(positions=positions, features=features) # Performs some data curation and general stuff common to any machine learning method.
		
		scaled_positions = positions.copy()
		for col in scaled_positions.columns:
			scaled_positions[col] = self.scalers[col].transform(scaled_positions[[col]])
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		self._svrs = {}
		for col in scaled_positions.columns:
			self._svrs[col] = SVR(kernel='rbf', C=1, epsilon=.1)
			self._svrs[col].fit(
				X = scaled_features,
				y = scaled_positions[col],
			)
	
	def reconstruct(self, features):
		super().reconstruct(features=features)
		
		reconstructed = pandas.DataFrame(index=features.index)
		for col in self._positions_columns:
			reconstructed[col] = self._svrs[col].predict(X = features)
			reconstructed[col] = self.scalers[col].inverse_transform(reconstructed[[col]])
		return reconstructed

def reconstruction_experiment(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	with bureaucrat.handle_task('reconstruction_experiment') as employee:
		if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
			raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
		flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
		parsed_from_waveforms = load_whole_dataframe(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
		
		positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
		positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
		for _ in {'x','y'}: # Remove offset so (0,0) is the center...
			positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
		
		data = parsed_from_waveforms
		data.reset_index('n_waveform', drop=True, inplace=True)
		
		data = data.query('n_pulse==1')
		data.reset_index('n_pulse', drop=True, inplace=True)
		
		# Calculate some event-wise stuff:
		data = data.unstack('n_channel')
		for n_channel in data.columns.get_level_values('n_channel').drop_duplicates():
			data[('Time from CH1 (s)',n_channel)] = data[('t_20 (s)',n_channel)] - data[('t_50 (s)',1)]
			
			data[('Total collected charge (V s)',n_channel)] = data[[('Collected charge (V s)',_) for _ in data.columns.get_level_values('n_channel').drop_duplicates()]].sum(axis=1)
			data[('Charge shared fraction',n_channel)] = data[('Collected charge (V s)',n_channel)]/data[('Total collected charge (V s)',n_channel)]
			
			data[('Total amplitude (V)',n_channel)] = data[[('Amplitude (V)',_) for _ in data.columns.get_level_values('n_channel').drop_duplicates()]].sum(axis=1)
			data[('Amplitude shared fraction',n_channel)] = data[('Amplitude (V)',n_channel)]/data[('Total amplitude (V)',n_channel)]
		data = data.stack('n_channel') # Revert.
		
		# Calculate variables that could be used for the reconstruction:
		data = data.unstack('n_channel')
		variables = {}
		for _ in {'Amplitude'}:
			variables[f'f_{_.lower()}_horizontal'] = data[(f'{_} shared fraction',1)] + data[(f'{_} shared fraction',3)] - data[(f'{_} shared fraction',2)] - data[(f'{_} shared fraction',4)]
			variables[f'f_{_.lower()}_vertical'] = data[(f'{_} shared fraction',1)] + data[(f'{_} shared fraction',2)] - data[(f'{_} shared fraction',3)] - data[(f'{_} shared fraction',4)]
		for _,_2 in variables.items():
			_2.name = _
		variables = pandas.concat([item for _,item in variables.items()], axis=1)
		data = data.stack('n_channel') # Revert what I have done before.

		variables = variables.merge(positions_data[['x (m)','y (m)']], left_index=True, right_index=True)
		
		variables.dropna(inplace=True)
		
		position_variables = ['x (m)','y (m)']
		reconstruction_variables = sorted(set(variables.columns)-set(position_variables))
		scalers = {}
		for col in variables.columns:
			scalers[col] = MinMaxScaler(copy=False)
			scalers[col].fit(variables[[col]])
			variables[f'{col} scaled'] = scalers[col].transform(variables[[col]])
		
		n_triggers_per_position = max(set(variables.index.get_level_values('n_trigger'))) + 1
		training_data = variables.query(f'n_trigger < {int(n_triggers_per_position/2)}')
		testing_data = variables.query(f'n_trigger >= {int(n_triggers_per_position/2)}')
		
		FEATURE_VARIABLES = ['f_amplitude_horizontal scaled','f_amplitude_vertical scaled']
		
		reconstructor = SVMReconstructor()
		reconstructor.fit(
			positions = training_data[['x (m)','y (m)']],
			features = training_data[FEATURE_VARIABLES],
		)
		
		reconstructed = reconstructor.reconstruct(testing_data[FEATURE_VARIABLES])
		reconstructed.columns = [f'{_} reco' for _ in reconstructed.columns]
		
		reconstructed['reconstruction error (m)'] = sum([(reconstructed[f'{_} reco']-positions_data[_])**2 for _ in ['x (m)','y (m)']])**.5
		
		result = reconstructed.groupby('n_position').agg([numpy.nanmean,numpy.nanstd])
		result.columns = [' '.join(_) for _ in result.columns]
		
		result = result.merge(positions_data[['x (m)','y (m)','n_x','n_y']], left_index=True, right_index=True)
		
		result = pandas.pivot_table(
			data = result,
			values = result.columns,
			index = 'n_x',
			columns = 'n_y',
		)
		
		for col in {'reconstruction error (m) nanstd','reconstruction error (m) nanmean'}:
			df = result[col]
			df.set_index(pandas.Index(sorted(set(positions_data['x (m)']))[::-1]), inplace=True)
			df = df.T.set_index(pandas.Index(sorted(set(positions_data['y (m)']))), 'y (m)').T
			df.index.name = 'x (m)'
			df.columns.name = 'y (m)'
			df = df.T
			fig = px.imshow(
				df,
				title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
				aspect = 'equal',
				labels = dict(
					color = col,
				),
				origin = 'lower',
				zmin = 0,
				zmax = 55e-6 if 'nanstd' in col else None,
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}.html',
				include_plotlyjs = 'cdn',
			)
		
		xx,yy = numpy.meshgrid(sorted(set(positions_data['x (m)'])), sorted(set(positions_data['y (m)'])))
		fig, ax = plt.subplots()
		ax.quiver(
			xx.T*1e6,
			yy.T*1e6,
			numpy.flip((result['x (m)'] - result['x (m) reco nanmean']).to_numpy(), 0),
			numpy.flip((result['y (m)'] - result['y (m) reco nanmean']).to_numpy(), 0),
			scale_units = 'xy',
		)
		ax.set_aspect('equal')
		ax.set_xlabel('x (µm)')
		ax.set_ylabel('y (µm)')
		for fmt in {'png','pdf'}:
			plt.savefig(employee.path_to_directory_of_my_task/f'vector_plot.{fmt}')
		
		a
		

if __name__ == '__main__':
	import argparse
	from grafica.plotly_utils.utils import set_my_template_as_default
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	reconstruction_experiment(bureaucrat)

