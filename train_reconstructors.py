from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from grafica.plotly_utils.utils import set_my_template_as_default
import numpy
import utils
import pickle
import reconstructors

def train_reconstructors(bureaucrat:RunBureaucrat):
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
	# ~ data.reset_index('n_pulse', drop=True, inplace=True)
	
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
	
	amplitude_share_data = data[['Amplitude shared fraction']].unstack('n_channel')
	amplitude_share_data.columns = [' '.join([str(__) for __ in _]) for _ in amplitude_share_data.columns]
	amplitude_share_data = amplitude_share_data.merge(positions_data[POSITION_VARIABLES_NAMES], left_index=True, right_index=True)
	
	amplitude_share_data_for_discrete_MLE_algorithm = amplitude_share_data + numpy.random.randn(*amplitude_share_data.shape)/999999999 # This has to be added because otherwise it fails due to some algebra error. I think that this is because the amplitude share data is so good quality (in terms of the correlations between the different channels) that then it fails to invert some matrix, or something like this. Adding some noise fixes this.
	
	n_triggers_per_position = max(set(variables.index.get_level_values('n_trigger'))) + 1
	RECONSTRUCTORS_TO_TEST = [
		# ~ dict(
			# ~ reconstructor = reconstructors.SVMReconstructor(),
			# ~ training_data = variables.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = variables.query(f'n_trigger >= {int(n_triggers_per_position*1/3)}'),
			# ~ features_variables_names = ['f_amplitude_horizontal','f_amplitude_vertical'],
			# ~ reconstructor_name = 'SVR_reconstructor_with_f_amplitudes',
		# ~ ),
		# ~ dict(
			# ~ reconstructor = reconstructors.SVMReconstructor(),
			# ~ training_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			# ~ reconstructor_name = 'SVR_reconstructor_with_amplitudes',
		# ~ ),
		# ~ dict(
			# ~ reconstructor = reconstructors.DNNReconstructor(),
			# ~ training_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ testing_data = amplitude_data.query(f'n_trigger < {int(n_triggers_per_position*2/3)}'),
			# ~ features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			# ~ reconstructor_name = 'DNN_reconstructor_with_amplitudes',
		# ~ ),
		dict(
			reconstructor = reconstructors.LookupTableReconstructor(),
			training_data = amplitude_data,
			testing_data = amplitude_data,
			features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			reconstructor_name = 'lookup_table_reconstructor_with_amplitudes',
			reconstructor_reconstruct_kwargs = dict(
				batch_size = 11111,
			),
		),
		dict(
			reconstructor = reconstructors.DiscreteMLEReconstructor(),
			training_data = amplitude_data,
			testing_data = amplitude_data,
			features_variables_names = [f'Amplitude (V) {_}' for _ in [1,2,3,4]],
			reconstructor_name = 'discrete_MLE_reconstructor_with_amplitudes',
			reconstructor_reconstruct_kwargs = dict(
				batch_size = 11111,
			),
		),
		dict(
			reconstructor = reconstructors.LookupTableReconstructor(),
			training_data = amplitude_share_data,
			testing_data = amplitude_share_data,
			features_variables_names = [f'Amplitude shared fraction {_}' for _ in [1,2,3,4]],
			reconstructor_name = 'lookup_table_reconstructor_with_amplitudes_fraction',
			reconstructor_reconstruct_kwargs = dict(
				batch_size = 11111,
			),
		),
		dict(
			reconstructor = reconstructors.DiscreteMLEReconstructor(),
			training_data = amplitude_share_data_for_discrete_MLE_algorithm,
			testing_data = amplitude_share_data_for_discrete_MLE_algorithm,
			features_variables_names = [f'Amplitude shared fraction {_}' for _ in [1,2,3,4]],
			reconstructor_name = 'discrete_MLE_reconstructor_with_amplitudes_fraction',
			reconstructor_reconstruct_kwargs = dict(
				batch_size = 11111,
			),
		),
	]
	for stuff in RECONSTRUCTORS_TO_TEST:
		print(f'{repr(stuff["reconstructor_name"])}...')
		with bureaucrat.handle_task(stuff['reconstructor_name']) as employee:
			
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
				)
				fig.update_layout(
					title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.write_html(
					path_for_plots/f'{col}_contour.html',
					include_plotlyjs = 'cdn',
				)
			
			reconstructor = stuff['reconstructor']
			print(f'Training...')
			reconstructor.fit(
				positions = training_data[POSITION_VARIABLES_NAMES],
				features = training_data[stuff['features_variables_names']],
			)
			with open(employee.path_to_directory_of_my_task/'reconstructor.pickle', 'wb') as ofile:
				pickle.dump(reconstructor, ofile, pickle.HIGHEST_PROTOCOL)
			print(f'Reconstructing...')
			reconstructed = reconstructor.reconstruct(testing_data[stuff['features_variables_names']], **stuff['reconstructor_reconstruct_kwargs'])
			
			print('Analyzing and plotting...')
			
			reconstructed.columns = [f'{_} reco' for _ in reconstructed.columns]
			
			reconstructed['reconstruction error (m)'] = sum([(reconstructed[f'{_} reco']-positions_data[_])**2 for _ in POSITION_VARIABLES_NAMES])**.5
			
			result = reconstructed.groupby('n_position').agg([numpy.nanmean,numpy.nanstd])
			result.columns = [' '.join(_) for _ in result.columns]
			result.rename(
				columns = {
					'reconstruction error (m) nanstd': 'Reconstruction uncertainty (m)',
					'reconstruction error (m) nanmean': 'Reconstruction bias (m)',
				},
				inplace = True,
			)
			
			x_grid_size = numpy.absolute(numpy.diff(positions_data['x (m)'])).mean()
			y_grid_size = numpy.absolute(numpy.diff(positions_data['y (m)'])).mean()
			xy_grid_sampling_contribution_to_the_uncertainty = x_grid_size/12**.5 + y_grid_size/12**.5
			result['Reconstruction uncertainty (m)'] = (result['Reconstruction uncertainty (m)']**2 + xy_grid_sampling_contribution_to_the_uncertainty**2)**.5
			# ~ print(f'xy_grid_sampling_contribution_to_the_uncertainty = {xy_grid_sampling_contribution_to_the_uncertainty*1e6:.1f} µm')
			
			for col in ['Reconstruction uncertainty (m)','Reconstruction bias (m)']:
				fig = utils.plot_as_xy_heatmap(
					z = result[col],
					positions_data = positions_data,
					title = f'{col} with {stuff["reconstructor_name"]}<br><sup>{bureaucrat.run_name}</sup>',
					aspect = 'equal',
					origin = 'lower',
					zmin = 0,
					zmax = 33e-6 if 'nanstd' in col else 33e-6 if 'nanmean' in col else None,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{col}_heatmap.html',
					include_plotlyjs = 'cdn',
				)
				fig = utils.plot_as_xy_contour(
					z = result[col],
					positions_data = positions_data,
				)
				fig.update_layout(
					title = f'{col} with {stuff["reconstructor_name"]}<br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{col}_contour.html',
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
	train_reconstructors(bureaucrat)

