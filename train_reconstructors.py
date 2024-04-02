from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy
import utils
import pickle
import reconstructors
import logging

def calculate_features(data:pandas.DataFrame, is_this_for_training:bool=False):
	# Calculate some event-wise stuff:
	data = data.unstack('n_channel')
	for n_channel in data.columns.get_level_values('n_channel').drop_duplicates():
		data[('Total amplitude (V)',n_channel)] = data[[('Amplitude (V)',_) for _ in data.columns.get_level_values('n_channel').drop_duplicates()]].sum(axis=1)
		data[('ASF',n_channel)] = data[('Amplitude (V)',n_channel)]/data[('Total amplitude (V)',n_channel)]
	# Calculate whatever will be fed into the ML reconstructors:
	features = {}
	for _ in {'ASF'}:
		features[f'f_{_}_horizontal'] = data[(f'ASF',1)] + data[(f'ASF',3)] - data[(f'ASF',2)] - data[(f'ASF',4)]
		features[f'f_{_}_vertical'] = data[(f'ASF',1)] + data[(f'ASF',2)] - data[(f'ASF',3)] - data[(f'ASF',4)]
	for _,_2 in features.items():
		_2.name = _
	features = pandas.concat([item for _,item in features.items()], axis=1)
	# ~ data = data.stack('n_channel') # Revert what I have done before.
	
	amplitude_shared_fraction = data[['ASF']]
	amplitude_shared_fraction.columns = [f'ASF {n_channel}' for n_channel in amplitude_shared_fraction.columns.get_level_values('n_channel')]
	if is_this_for_training == True:
		amplitude_shared_fraction = amplitude_shared_fraction + numpy.random.randn(*amplitude_shared_fraction.shape)/999999 # This has to be added because otherwise it fails due to some algebra error. I think that this is because the amplitude share data is so good quality (in terms of the correlations between the different channels) that then it fails to invert some matrix, or something like this. Adding some noise fixes this.
	
	amplitude = data[['Amplitude (V)']]
	amplitude.columns = [' '.join([str(__) for __ in _]) for _ in amplitude.columns]
	
	return pandas.concat([features,amplitude_shared_fraction,amplitude], axis=1)


#HERE is my new func
def features_parisa(data:pandas.DataFrame):
	my_data = data.copy()
	my_data = my_data.unstack('n_channel')
	ChA = my_data['Amplitude (V)'] > 0.01
	data_t_50 = my_data[['t_50 (s)']][ChA]
	ChA = ChA.astype(float)
	ChA.columns = [f'ChA_{n}' for n in ChA.columns]
	delta_t_12 =  data_t_50[('t_50 (s)',1)] - data_t_50[('t_50 (s)',2)]
	delta_t_13 =  data_t_50[('t_50 (s)',1)] - data_t_50[('t_50 (s)',3)]
	delta_t_14 =  data_t_50[('t_50 (s)',1)] - data_t_50[('t_50 (s)',4)]
	delta_t_23 =  data_t_50[('t_50 (s)',2)] - data_t_50[('t_50 (s)',3)]
	delta_t_24 =  data_t_50[('t_50 (s)',2)] - data_t_50[('t_50 (s)',4)]
	delta_t_34 =  data_t_50[('t_50 (s)',3)] - data_t_50[('t_50 (s)',4)]
	delta_t = pandas.DataFrame( {'delta_t_12': delta_t_12, 'delta_t_13': delta_t_13, 'delta_t_14': delta_t_14, 'delta_t_23': delta_t_23, 'delta_t_24': delta_t_24, 'delta_t_34': delta_t_34 }	)
	delta_t = delta_t.fillna(value = 0)
	feature = pandas.concat([delta_t, ChA ], axis = 'columns' ) 
	#print(feature.query('ChA_1 == 1 and ChA_2 ==1 and ChA_3 ==1'))
	#exit()
	return feature



def train_DNN_using_t50(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	# Load data:
	if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
		raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
	logging.info(f'Loading data from {bureaucrat.run_name}')
	flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
	connection = sqlite3.connect(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
	data = pandas.read_sql("SELECT n_position,n_trigger,n_channel,`t_50 (s)`,`Amplitude (V)` FROM dataframe_table WHERE n_pulse==1", connection)
	data.set_index(['n_position','n_trigger','n_channel'], inplace=True)
	
	positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	
	# ~ data = data.unstack('n_channel')
	# ~ data.columns = [' '.join([str(_) for _ in col]) for col in data]
	# ~ data = data.merge(positions_data[['x (m)','y (m)']], left_index=True, right_index=True)
	# ~ data = data.stack('n_channel')
	
	features = features_parisa(data)
	features = features.merge(positions_data[['x (m)','y (m)']], left_index=True, right_index=True)
	with bureaucrat.handle_task('train_DNN_using_t50') as employee:
		reconstructor = reconstructors.DNNPositionReconstructor()
		logging.info(f'Training DNN...')
		print(features)
		reconstructor.fit(
			features = features[['delta_t_12','delta_t_13','delta_t_14','delta_t_23','delta_t_24','delta_t_34','ChA_1','ChA_2','ChA_3','ChA_4']],
			positions = features[['x (m)','y (m)']],
			batch_size = 9999,
		)
		with open(employee.path_to_directory_of_my_task/'reconstructor.pickle', 'wb') as ofile:
			pickle.dump(reconstructor, ofile, pickle.HIGHEST_PROTOCOL)

	logging.info(f'Finished with t_50 neural network')	
		
		
		
		
		
		
def train_reconstructors(bureaucrat:RunBureaucrat, split_into_n_regions:int):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	POSITION_VARIABLES_NAMES = ['x (m)','y (m)']
	
	# Load data:
	if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
		raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
	logging.info(f'Loading data from {bureaucrat.run_name}')
	flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
	connection = sqlite3.connect(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
	data = pandas.read_sql("SELECT n_position,n_trigger,n_channel,`Amplitude (V)` FROM dataframe_table WHERE n_pulse==1", connection)
	data.set_index(['n_position','n_trigger','n_channel'], inplace=True)
	
	positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	#I can ignore this part
	pads_arrangement = pandas.read_csv(
		bureaucrat.path_to_run_directory/'pads_arrangement.csv',
		index_col = 'n_channel',
		dtype = {
			'n_channel': int,
			'n_row': int,
			'n_col': int,
		},
	)
	
	positions_data, n_position_mapping = utils.resample_positions(positions_data,*[split_into_n_regions for _ in ['x','y']])
	
	features = calculate_features(data, is_this_for_training=True)
	
	# Update position data:
	features = features.join(n_position_mapping, on='n_position')
	features.reset_index('n_position', inplace=True, drop=True)
	features.reset_index(inplace=True, drop=False)
	features.set_index('n_position', inplace=True)
	for n_position, df in features.groupby('n_position'):
		features.loc[n_position,'n_trigger'] = numpy.arange(len(df))
	features.reset_index(inplace=True, drop=False)
	features.set_index(['n_position','n_trigger'], inplace=True)
	
	features = features.merge(positions_data[POSITION_VARIABLES_NAMES], left_index=True, right_index=True)
	
	n_channels = data.index.get_level_values('n_channel').drop_duplicates()
	amplitude_data_for_reconstructors = features[[f'Amplitude (V) {n_channel}' for n_channel in n_channels] + POSITION_VARIABLES_NAMES]
	amplitude_share_data_for_reconstructors = features[[f'ASF {n_channel}' for n_channel in n_channels] + POSITION_VARIABLES_NAMES]
	RECONSTRUCTORS_TO_TEST = [
		dict(
			reconstructor = reconstructors.DNNPositionReconstructor(),
			training_data = amplitude_share_data_for_reconstructors,
			testing_data = amplitude_share_data_for_reconstructors,
			features_variables_names = sorted(set(amplitude_share_data_for_reconstructors.columns).difference(POSITION_VARIABLES_NAMES)),
			reconstructor_name = 'DNNPositionReconstructor_using_ASF',
			reconstructor_fit_kwargs = dict(
				batch_size = 11111,
			),
			reconstructor_reconstruct_kwargs = dict(
				batch_size = 11111,
			),
		),
	]
	for stuff in RECONSTRUCTORS_TO_TEST:
		with bureaucrat.handle_task(f"position_reconstructor_{stuff['reconstructor_name'].replace(' ','')}_{split_into_n_regions}x{split_into_n_regions}") as employee:
			
			training_data = stuff['training_data'].dropna()
			testing_data = stuff['testing_data'].dropna()
			
			path_for_plots = employee.path_to_directory_of_my_task/'features'
			for col in stuff['features_variables_names']:
				fig = utils.plot_as_xy_heatmap(
					z = training_data.groupby('n_position').agg(numpy.nanmean)[col],
					positions_data = positions_data,
					title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
					aspect = 'equal',
					origin = 'lower',
				)
				path_for_plots.mkdir(exist_ok=True)
				fig.write_html(
					path_for_plots/f'{col}_heatmap.html',
					include_plotlyjs = 'cdn',
				)
				fig = utils.plot_as_xy_contour(
					z = training_data.groupby('n_position').agg(numpy.nanmean)[col],
					positions_data = positions_data,
					smoothing_sigma = 2,
				)
				fig.update_layout(
					title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.write_html(
					path_for_plots/f'{col}_contour.html',
					include_plotlyjs = 'cdn',
				)
			
			reconstructor = stuff['reconstructor']
			logging.info(f'Training {repr(stuff["reconstructor_name"])}...')
			reconstructor.fit(
				positions = training_data[POSITION_VARIABLES_NAMES],
				features = training_data[stuff['features_variables_names']],
				**stuff['reconstructor_fit_kwargs'],
			)
			with open(employee.path_to_directory_of_my_task/'reconstructor.pickle', 'wb') as ofile:
				pickle.dump(reconstructor, ofile, pickle.HIGHEST_PROTOCOL)

		logging.info(f'Finished with {repr(stuff["reconstructor_name"])}')

if __name__ == '__main__':
	import argparse
	from plotly_utils import set_my_template_as_default
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(asctime)s|%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = Path,
	)
	args = parser.parse_args()
	
	train_DNN_using_t50(bureaucrat=RunBureaucrat(args.directory))
