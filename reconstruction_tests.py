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
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

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
			self._svrs[col] = SVR(kernel='rbf', C=100, epsilon=.1)
			self._svrs[col].fit(
				X = scaled_features,
				y = scaled_positions[col],
			)
		
	def reconstruct(self, features):
		super().reconstruct(features=features)
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		reconstructed = pandas.DataFrame(index=features.index)
		for col in self._positions_columns:
			reconstructed[col] = self._svrs[col].predict(X = scaled_features)
			reconstructed[col] = self.scalers[col].inverse_transform(reconstructed[[col]])
		return reconstructed

class DNNReconstructor(RSDReconstructor):
	class RSDDataLoader(Dataset):
		def __init__(self, positions:pandas.DataFrame, features:pandas.DataFrame, transform=None, target_transform=None, batch_size:int=1, shuffle:bool=False):
			df = pandas.concat([positions,features], axis=1)
			if shuffle == True:
				df = df.sample(frac=1)
			
			self.positions = torch.tensor(df[positions.columns].values).to(torch.float32)
			self.features = torch.tensor(df[features.columns].values).to(torch.float32)
			
			self.transform = transform
			self.target_transform = target_transform
			
			self.batch_size = batch_size

		def __len__(self):
			return -(len(self.positions)//-self.batch_size)
		
		def __getitem__(self, idx):
			bs = self.batch_size
			if (idx+1)*bs < len(self.positions):
				position = self.positions[idx*bs:(idx+1)*bs,:]
				features = self.features[idx*bs:(idx+1)*bs,:]
			elif idx*bs < len(self.positions):
				position = self.positions[idx*bs:,:]
				features = self.features[idx*bs:,:]
			else:
				raise StopIteration()
			if self.transform:
				position = self.transform(position)
			if self.target_transform:
				features = self.target_transform(features)
			return features, position
	
	def __init__(self):
		class NeuralNetwork(nn.Module):
			def __init__(self):
				super().__init__()
				self.flatten = nn.Flatten()
				self.linear_relu_stack = nn.Sequential(
					nn.Linear(4, 11),
					nn.ReLU(),
					nn.Linear(11, 11),
					nn.ReLU(),
					nn.Linear(11, 11),
					nn.ReLU(),
					nn.Linear(11, 2)
				)

			def forward(self, x):
				x = self.flatten(x)
				logits = self.linear_relu_stack(x)
				return logits
		
		device = (
			"cuda"
			if torch.cuda.is_available()
			else "mps"
			if torch.backends.mps.is_available()
			else "cpu"
		)
		self.device = device
		
		self.dnn = NeuralNetwork().to(device)
		
	def fit(self, positions, features):
		def train(dataloader, model, loss_fn, optimizer):
			model.train()
			for batch, (X, y) in enumerate(dataloader):
				X, y = X.to(self.device), y.to(self.device)
				
				# Compute prediction error
				pred = model(X)
				loss = loss_fn(pred, y)

				# Backpropagation
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				
				# ~ print(f"loss: {loss.item():>7f}  [batch: {batch}/{len(dataloader)-1}]")
		
		super().fit(positions=positions, features=features) # Performs some data curation and general stuff common to any machine learning method.
		
		# Scale between 0 and 1:
		scaled_positions = positions.copy()
		for col in scaled_positions.columns:
			scaled_positions[col] = self.scalers[col].transform(scaled_positions[[col]])
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		for t in range(555):
			print(f'EPOCH {t}')
			train(
				dataloader = self.RSDDataLoader(
					positions = scaled_positions,
					features = scaled_features,
					batch_size = int(len(scaled_positions)/11),
					shuffle = True,
				),
				model = self.dnn, 
				loss_fn = nn.CrossEntropyLoss(),
				optimizer = torch.optim.SGD(self.dnn.parameters(), lr=1e-3),
			)
	
	def reconstruct(self, features):
		super().reconstruct(features=features)
		
		def predict(dataloader, model, loss_fn):
			print(f'Testing...')
			
			return prediction
		
		# Scale between 0 and 1:
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		dataloader = self.RSDDataLoader(
			positions = pandas.DataFrame(index=features.index, data=numpy.zeros((len(features),len(self._positions_columns))), columns=self._positions_columns), # Create fake data just because this requires it, but it will not be used...
			features = scaled_features,
			batch_size = len(features),
		)
		self.dnn.eval()
		with torch.no_grad():
			for X, y in dataloader:
				X = X.to(self.device)
				prediction = self.dnn(X)
		
		reconstructed = pandas.DataFrame(
			index = features.index,
			data = prediction.numpy(),
			columns = self._positions_columns,
		)
		for col in reconstructed:
			reconstructed[col] = self.scalers[col].inverse_transform(reconstructed[[col]])
		
		return reconstructed

class LookupTableReconstructor(RSDReconstructor):
	def fit(self, positions, features):
		super().fit(positions=positions, features=features) # Performs some data curation and general stuff common to any method.
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		self.lookup_table_of_scaled_features = scaled_features.groupby('n_position').agg(numpy.nanmean)
		self.lookup_table_of_positions = positions.groupby('n_position').agg(numpy.nanmean)
		
	def reconstruct(self, features):
		super().reconstruct(features=features)
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		distances = []
		for feature in self._features_columns:
			_ = numpy.subtract.outer(
				scaled_features[feature].to_numpy(),
				self.lookup_table_of_scaled_features[feature].to_numpy(),
			)
			_ **= 2
			distances.append(_)
		distances = sum(distances)
		idx_reconstructed = numpy.argmin(distances, axis=1)
		
		reconstructed_positions = pandas.DataFrame(
			data = self.lookup_table_of_positions.iloc[idx_reconstructed].to_numpy(),
			index = features.index,
			columns = self._positions_columns,
		)
		return reconstructed_positions

def reconstruction_experiment(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	POSITION_VARIABLES_NAMES = ['x (m)','y (m)']
	
	# Load data:
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
	
	pads_arrangement = pandas.read_csv(
		bureaucrat.path_to_run_directory/'pads_arrangement.csv',
		index_col = 'n_channel',
		dtype = {
			'n_channel': int,
			'n_row': int,
			'n_col': int,
		},
	)
	
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

	variables = variables.merge(positions_data[POSITION_VARIABLES_NAMES], left_index=True, right_index=True)
	
	amplitude_data = data[['Amplitude (V)']].unstack('n_channel')
	amplitude_data.columns = [' '.join([str(__) for __ in _]) for _ in amplitude_data.columns]
	amplitude_data = amplitude_data.merge(positions_data[POSITION_VARIABLES_NAMES], left_index=True, right_index=True)
	
	n_triggers_per_position = max(set(variables.index.get_level_values('n_trigger'))) + 1
	RECONSTRUCTORS_TO_TEST = [
		# ~ dict(
			# ~ reconstructor = SVMReconstructor(),
			# ~ training_data = variables.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = variables.query(f'n_trigger >= {int(n_triggers_per_position*1/3)}'),
			# ~ features_variables_names = ['f_amplitude_horizontal','f_amplitude_vertical'],
			# ~ reconstructor_name = 'SVR_reconstruction_test',
		# ~ ),
		# ~ dict(
			# ~ reconstructor = SVMReconstructor(),
			# ~ training_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			# ~ reconstructor_name = 'SVR_reconstruction_with_amplitudes',
		# ~ ),
		# ~ dict(
			# ~ reconstructor = DNNReconstructor(),
			# ~ training_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			# ~ reconstructor_name = 'DNN_reconstruction_with_amplitudes',
		# ~ ),
		dict(
			reconstructor = LookupTableReconstructor(),
			training_data = amplitude_data,
			testing_data = amplitude_data.query(f'n_trigger < 7'),
			features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			reconstructor_name = 'lookup_table_reconstruction_with_amplitudes',
		),
	]
	for stuff in RECONSTRUCTORS_TO_TEST:
		print(f'{repr(stuff["reconstructor_name"])}...')
		with bureaucrat.handle_task(stuff['reconstructor_name']) as employee:
			
			training_data = stuff['training_data'].dropna()
			testing_data = stuff['testing_data'].dropna()
			
			reconstructor = stuff['reconstructor']
			print(f'Training...')
			reconstructor.fit(
				positions = training_data[POSITION_VARIABLES_NAMES],
				features = training_data[stuff['features_variables_names']],
			)
			print(f'Reconstructing...')
			reconstructed = reconstructor.reconstruct(testing_data[stuff['features_variables_names']])
			
			print('Analyzing and plotting...')
			
			reconstructed.columns = [f'{_} reco' for _ in reconstructed.columns]
			
			reconstructed['reconstruction error (m)'] = sum([(reconstructed[f'{_} reco']-positions_data[_])**2 for _ in POSITION_VARIABLES_NAMES])**.5
			
			result = reconstructed.groupby('n_position').agg([numpy.nanmean,numpy.nanstd])
			result.columns = [' '.join(_) for _ in result.columns]
			
			for col in stuff['features_variables_names']:
				fig = utils.plot_as_xy_heatmap(
					z = training_data.groupby('n_position').agg(numpy.nanmean)[col],
					positions_data = positions_data,
					title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
					aspect = 'equal',
					origin = 'lower',
				)
				path_for_plots = employee.path_to_directory_of_my_task/'features'
				path_for_plots.mkdir(exist_ok=True)
				fig.write_html(
					path_for_plots/f'{col}.html',
					include_plotlyjs = 'cdn',
				)
			
			for col in {'reconstruction error (m) nanstd','reconstruction error (m) nanmean'}:
				fig = utils.plot_as_xy_contour(
					z = result[col],
					positions_data = positions_data,
					title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
					# ~ aspect = 'equal',
					# ~ origin = 'lower',
					# ~ zmin = 0,
					# ~ zmax = 33e-6 if 'nanstd' in col else 33e-6 if 'nanmean' in col else None,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{col}.html',
					include_plotlyjs = 'cdn',
				)
			
			
			z = result.copy()
			z = z.merge(positions_data[['x (m)','y (m)','n_x','n_y']], left_index=True, right_index=True)
			z = pandas.pivot_table(
				data = z,
				values = z.columns,
				index = 'n_x',
				columns = 'n_y',
			)
			xx,yy = numpy.meshgrid(sorted(set(positions_data['x (m)'])), sorted(set(positions_data['y (m)'])))
			fig, ax = plt.subplots()
			ax.quiver(
				xx.T*1e6,
				yy.T*1e6,
				numpy.flip((z['x (m)'] - z['x (m) reco nanmean']).to_numpy(), 0),
				numpy.flip((z['y (m)'] - z['y (m) reco nanmean']).to_numpy(), 0),
				angles = 'xy', 
				scale_units = 'xy', 
				scale = 1e-6,
			)
			ax.set_aspect('equal')
			ax.set_xlabel('x (µm)')
			ax.set_ylabel('y (µm)')
			plt.title(f'Reconstruction bias plot\n{bureaucrat.run_name}')
			for fmt in {'png','pdf'}:
				plt.savefig(employee.path_to_directory_of_my_task/f'vector_plot.{fmt}')
		
		print(f'Finished {repr(stuff["reconstructor_name"])}.')

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

