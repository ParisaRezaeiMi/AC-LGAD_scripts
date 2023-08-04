import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from scipy.stats import gaussian_kde
import warnings

class RSDPositionReconstructor:
	def fit(self, positions, features):
		"""Fit the model using the training data.
		
		Arguments
		---------
		positions: array like
			An `n_events`×`2` array containing as columns the positions
			in x and y.
		features: array like
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
		
		self.positions_names = positions.columns
		self.features_names = features.columns
		
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
		if set(features.columns) != set(self.features_names):
			raise ValueError(f'`features` columns do not match the ones used during the fitting. During the fitting process I received {sorted(set(self.features_names))} and now for reconstruction I am receiving {sorted(set(features.columns))}. ')

class SVMPositionReconstructor(RSDPositionReconstructor):
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
		for col in self.positions_names:
			reconstructed[col] = self._svrs[col].predict(X = scaled_features)
			reconstructed[col] = self.scalers[col].inverse_transform(reconstructed[[col]])
		return reconstructed

class DNNPositionReconstructor(RSDPositionReconstructor):
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
			positions = pandas.DataFrame(index=features.index, data=numpy.zeros((len(features),len(self.positions_names))), columns=self.positions_names), # Create fake data just because this requires it, but it will not be used...
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
			columns = self.positions_names,
		)
		for col in reconstructed:
			reconstructed[col] = self.scalers[col].inverse_transform(reconstructed[[col]])
		
		return reconstructed

class LookupTablePositionReconstructor(RSDPositionReconstructor):
	def fit(self, positions, features):
		super().fit(positions=positions, features=features) # Performs some data curation and general stuff common to any method.
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		self.lookup_table_of_scaled_features = scaled_features.groupby('n_position').agg(numpy.nanmean)
		self.lookup_table_of_positions = positions.groupby('n_position').agg(numpy.nanmean)
		
	def reconstruct(self, features, batch_size:int=None):
		"""
		Arguments
		---------
		features: pandas.DataFrame
			See description of `RSDPositionReconstructor`.
		batch_size: int, default `None`
			When `features` is very large, it can cause the computer to
			run out of memory when performing the reconstruction. The
			`features` data frame can be split in smaller batches, each
			with a number of events (rows) given by `batch_size`. If `batch_size`
			is `None` then no splitting is performed.
		"""
		super().reconstruct(features=features)
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		if batch_size is None or len(features)<batch_size:
			reconstructed_positions = self._reconstruct_using_scaled_features(scaled_features)
		else:
			reconstructed_positions = []
			for scaled_features_batch in numpy.array_split(scaled_features, int(len(scaled_features)/batch_size)):
				_ = self._reconstruct_using_scaled_features(scaled_features_batch)
				reconstructed_positions.append(_)
			reconstructed_positions = pandas.concat(reconstructed_positions)
		
		return reconstructed_positions
	
	def _reconstruct_using_scaled_features(self, scaled_features):
		"""Performs the actual reconstruction process using the scaled
		features. Returns the reconstructed positions."""
		distances = []
		for feature in self.features_names:
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
			index = scaled_features.index,
			columns = self.positions_names,
		)
		return reconstructed_positions

class DiscreteMLEPositionReconstructor(RSDPositionReconstructor):
	def fit(self, positions, features):
		super().fit(positions=positions, features=features) # Performs some data curation and general stuff common to any method.
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		KDEs = []
		n_positions = []
		for n_position, this_position_scaled_features in scaled_features.groupby('n_position'):
			this_position_KDE = gaussian_kde(this_position_scaled_features.values.transpose())
			KDEs.append(this_position_KDE)
			n_positions.append(n_position)
		self.scaled_features_KDEs = pandas.Series(
			data = KDEs,
			index = pandas.Index(
				data = n_positions,
				name = 'n_position',
			),
			name = 'scaled_features_KDEs',
		)
		
		self.lookup_table_of_positions = positions.groupby('n_position').agg(numpy.nanmean)
	
	def reconstruct(self, features, batch_size:int=None):
		"""
		Arguments
		---------
		features: pandas.DataFrame
			See description of `RSDPositionReconstructor`.
		batch_size: int, default `None`
			When `features` is very large, it can cause the computer to
			run out of memory when performing the reconstruction. The
			`features` data frame can be split in smaller batches, each
			with a number of events (rows) given by `batch_size`. If `batch_size`
			is `None` then no splitting is performed.
		"""
		super().reconstruct(features=features)
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
		if batch_size is None or len(features)<batch_size:
			reconstructed_positions = self._reconstruct_using_scaled_features(scaled_features)
		else:
			reconstructed_positions = []
			for scaled_features_batch in numpy.array_split(scaled_features, int(len(scaled_features)/batch_size)):
				_ = self._reconstruct_using_scaled_features(scaled_features_batch)
				reconstructed_positions.append(_)
			reconstructed_positions = pandas.concat(reconstructed_positions)
		
		return reconstructed_positions
	
	def _reconstruct_using_scaled_features(self, scaled_features):
		"""Performs the actual reconstruction process using the scaled
		features. Returns the reconstructed positions."""
		likelihoods = []
		n_positions = []
		for n_position, KDE in self.scaled_features_KDEs.items():
			likelihood = KDE(scaled_features.values.transpose())
			likelihoods.append(likelihood)
			n_positions.append(n_position)
		likelihoods = numpy.array(likelihoods)
		n_positions = numpy.array(n_positions)
		most_likely_n_position = numpy.argmax(likelihoods, axis=0)
		reconstructed_positions = self.lookup_table_of_positions.loc[most_likely_n_position]
		reconstructed_positions.index = scaled_features.index
		
		return reconstructed_positions

########################################################################
########################################################################
########################################################################

class RSDTimeReconstructor:
	def reconstruct(self, features):
		raise NotImplemented(f'This is a prototype method, do not call it.')

class OnePadTimeReconstructor(RSDTimeReconstructor):
	def reconstruct(self, features):
		"""Reconstruct the impact time for each event.
		
		Arguments
		---------
		features: pandas.DataFrame
			A data frame with the data required for the time reconstruction.
			A data frame of the form
			```
												  time                                              weight                              
			n_channel                                1             2             3             4         1         2         3         4
			n_position n_trigger n_pulse                                                                                                
			0          0         1        2.553036e-09  2.226954e-09  3.443067e-09  3.550194e-09  0.015508  0.092558  0.006624  0.007211
								 2        1.011417e-07  1.008971e-07  9.415563e-08  6.134500e-08  0.012811  0.061471  0.005120  0.004641
					   1         1        2.594698e-09  2.219940e-09  3.370886e-09  3.225759e-09  0.013190  0.083752  0.006920  0.006954
								 2        1.012139e-07  1.008638e-07  4.050543e-08  4.041096e-08  0.009761  0.057924  0.008160  0.007454
					   2         1        2.573569e-09  2.207249e-09  3.564645e-09  3.413975e-09  0.016862  0.102959  0.007694  0.006463
			...                                    ...           ...           ...           ...       ...       ...       ...       ...
			2024       8         2        1.021516e-07  6.929512e-08  4.686199e-08  1.008832e-07  0.008351  0.005368  0.004110  0.026114
					   9         1        2.919888e-08  1.108853e-08  2.850360e-08  2.259533e-09  0.006050  0.008061  0.005201  0.021381
								 2        8.119503e-08  9.136477e-08  1.023884e-07  1.009116e-07  0.006302  0.005466  0.005987  0.013988
					   10        1        7.226838e-10  2.579804e-08  2.914910e-09  2.216060e-09  0.004999  0.005680  0.005031  0.023300
								 2        9.012005e-08  1.187095e-07  3.409594e-08  1.009591e-07  0.004036  0.004951  0.004300  0.020847

			[44550 rows x 8 columns]
			```
			is expected.
		"""
		if not isinstance(features, pandas.DataFrame):
			raise TypeError(f'`features` must be an instance of {pandas.DataFrame}, received object of type {type(features)}. ')
		if set(features.columns.get_level_values(0)) != {'weight','time'} or len(features.columns.names) != 2 or features.columns.names[1] != 'n_channel':
			raise ValueError(f'`features` must have 2 levels of columns, the top level should be `weight` and `time`, while the second level must be named `n_channel` and should contain integer numbers for each of the channels.')
		
		active_channel = features[['weight']]
		_ = active_channel.max(axis=1)
		for col in active_channel.columns:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				active_channel[col] = active_channel[col] == _
		active_channel = active_channel.stack('n_channel')['weight']
		active_channel.name = 'channel_is_active'
		
		reconstructed_time = features.stack('n_channel').loc[active_channel,'time']
		reconstructed_time.reset_index('n_channel', drop=True, inplace=True)
		
		return reconstructed_time

class MultipadWeightedTimeReconstructor(RSDTimeReconstructor):
	def reconstruct(self, features):
		"""Reconstruct the impact time for each event.
		
		Arguments
		---------
		features: pandas.DataFrame
			A data frame with the data required for the time reconstruction.
			A data frame of the form
			```
												  time                                              weight                              
			n_channel                                1             2             3             4         1         2         3         4
			n_position n_trigger n_pulse                                                                                                
			0          0         1        2.553036e-09  2.226954e-09  3.443067e-09  3.550194e-09  0.015508  0.092558  0.006624  0.007211
								 2        1.011417e-07  1.008971e-07  9.415563e-08  6.134500e-08  0.012811  0.061471  0.005120  0.004641
					   1         1        2.594698e-09  2.219940e-09  3.370886e-09  3.225759e-09  0.013190  0.083752  0.006920  0.006954
								 2        1.012139e-07  1.008638e-07  4.050543e-08  4.041096e-08  0.009761  0.057924  0.008160  0.007454
					   2         1        2.573569e-09  2.207249e-09  3.564645e-09  3.413975e-09  0.016862  0.102959  0.007694  0.006463
			...                                    ...           ...           ...           ...       ...       ...       ...       ...
			2024       8         2        1.021516e-07  6.929512e-08  4.686199e-08  1.008832e-07  0.008351  0.005368  0.004110  0.026114
					   9         1        2.919888e-08  1.108853e-08  2.850360e-08  2.259533e-09  0.006050  0.008061  0.005201  0.021381
								 2        8.119503e-08  9.136477e-08  1.023884e-07  1.009116e-07  0.006302  0.005466  0.005987  0.013988
					   10        1        7.226838e-10  2.579804e-08  2.914910e-09  2.216060e-09  0.004999  0.005680  0.005031  0.023300
								 2        9.012005e-08  1.187095e-07  3.409594e-08  1.009591e-07  0.004036  0.004951  0.004300  0.020847

			[44550 rows x 8 columns]
			```
			is expected.
		"""
		if not isinstance(features, pandas.DataFrame):
			raise TypeError(f'`features` must be an instance of {pandas.DataFrame}, received object of type {type(features)}. ')
		if set(features.columns.get_level_values(0)) != {'weight','time'} or len(features.columns.names) != 2 or features.columns.names[1] != 'n_channel':
			raise ValueError(f'`features` must have 2 levels of columns, the top level should be `weight` and `time`, while the second level must be named `n_channel` and should contain integer numbers for each of the channels.')
		
		leading_channel = features[['weight']]
		_ = leading_channel.max(axis=1)
		for col in leading_channel.columns:
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				leading_channel[col] = leading_channel[col] == _
		leading_channel = leading_channel.stack('n_channel')['weight']
		leading_channel.name = 'channel_is_active'
		
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			time_from_leading_channel = features[['time']].stack('n_channel')
			time_from_leading_channel.loc[~leading_channel] = float('NaN')
			time_from_leading_channel.reset_index('n_channel', drop=True, inplace=True)
			time_from_leading_channel = time_from_leading_channel.groupby(time_from_leading_channel.index.names).agg(numpy.nansum)
			time_from_leading_channel = time_from_leading_channel['time']
			_ = time_from_leading_channel
			time_from_leading_channel = features[['time']]
			for n_channel in features.columns.get_level_values('n_channel').drop_duplicates():
				time_from_leading_channel[('time',n_channel)] -= _
			time_from_leading_channel = time_from_leading_channel.stack('n_channel')['time']
			time_from_leading_channel[(time_from_leading_channel.abs()>100e-12)] = float('NaN')
			time_from_leading_channel.name = ''
			time_from_leading_channel.loc[~time_from_leading_channel.isna()] = 1
			time_from_leading_channel = time_from_leading_channel.unstack('n_channel')
			use_these_for_computing_time_resolution = time_from_leading_channel
		
		reconstructed_time = (features['time']*features['weight']*use_these_for_computing_time_resolution).sum(axis=1, skipna=True)/(features['weight']*use_these_for_computing_time_resolution).sum(axis=1, skipna=True)
		return reconstructed_time
