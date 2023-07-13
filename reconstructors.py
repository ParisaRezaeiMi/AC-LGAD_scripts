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

class DiscreteMLEReconstructor(RSDReconstructor):
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
	
	def reconstruct(self, features):
		super().reconstruct(features=features)
		
		scaled_features = features.copy()
		for feature in scaled_features.columns:
			scaled_features[feature] = self.scalers[feature].transform(scaled_features[[feature]])
		
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
		reconstructed_positions.index = features.index
		
		return reconstructed_positions
