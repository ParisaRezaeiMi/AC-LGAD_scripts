from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import pickle
import reconstructors
import utils
import numpy
import matplotlib.pyplot as plt

def reconstruct(bureaucrat:RunBureaucrat, path_to_reconstructor_pickle:Path):
	POSITION_VARIABLES_NAMES = ['x (m)','y (m)']
	
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	reconstructor_bureaucrat = RunBureaucrat(path_to_reconstructor_pickle.parent.parent)
	reconstructor_name = path_to_reconstructor_pickle.parent.parts[-1]
	reconstructor_bureaucrat.check_these_tasks_were_run_successfully(reconstructor_name)
	
	with open(path_to_reconstructor_pickle, 'rb') as ifile:
		reconstructor = pickle.load(ifile)
	
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
	
	# ~ amplitude_share_data_for_discrete_MLE_algorithm = amplitude_share_data + numpy.random.randn(*amplitude_share_data.shape)/999999999 # This has to be added because otherwise it fails due to some algebra error. I think that this is because the amplitude share data is so good quality (in terms of the correlations between the different channels) that then it fails to invert some matrix, or something like this. Adding some noise fixes this.
	
	with bureaucrat.handle_task(f'reconstruction_using_{reconstructor_name}') as employee:
		with open(employee.path_to_directory_of_my_task/'readme.txt', 'w') as ofile:
			print(f'This reconstruction was performed using the following reconstructor:', file=ofile)
			print(f'- {reconstructor_name}', file=ofile)
			print(f'- {path_to_reconstructor_pickle.resolve()}', file=ofile)
		
		testing_data = amplitude_data
		
		path_for_plots = employee.path_to_directory_of_my_task/'features'
		for col in reconstructor.features_names:
			fig = utils.plot_as_xy_heatmap(
				z = testing_data.groupby('n_position').agg(numpy.nanmean)[col],
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
				z = testing_data.groupby('n_position').agg(numpy.nanmean)[col],
				positions_data = positions_data,
			)
			fig.update_layout(
				title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				path_for_plots/f'{col}_contour.html',
				include_plotlyjs = 'cdn',
			)
		
		print(f'Reconstructing...')
		reconstructed = reconstructor.reconstruct(
			testing_data[reconstructor.features_names],
			batch_size = 11111,
		)
		
		print('Analyzing and plotting...')
		
		reconstructed.columns = [f'{_} reco' for _ in reconstructed.columns]
		
		utils.save_dataframe(
			reconstructed,
			name = 'reconstructed_positions',
			location = employee.path_to_directory_of_my_task,
		)
		
		reconstructed['reconstruction error (m)'] = sum([(reconstructed[f'{_} reco']-positions_data[_])**2 for _ in POSITION_VARIABLES_NAMES])**.5
		
		result = reconstructed.groupby('n_position').agg([numpy.nanmean,numpy.nanstd])
		result.columns = [' '.join(_) for _ in result.columns]
		
		reconstruction_error = pandas.Series(
			data = (result['reconstruction error (m) nanstd']**2 + result['reconstruction error (m) nanmean']**2)**.5,
			index = result.index,
			name = 'Reconstruction error (m)',
		)
		
		for col in ['reconstruction error (m) nanstd','reconstruction error (m) nanmean']:
			fig = utils.plot_as_xy_heatmap(
				z = result[col],
				positions_data = positions_data,
				title = f'{col} with {reconstructor_name}<br><sup>{bureaucrat.run_name}</sup>',
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
				title = f'{col} with {reconstructor_name}<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_contour.html',
				include_plotlyjs = 'cdn',
			)
		
		fig = utils.plot_as_xy_heatmap(
			z = reconstruction_error,
			positions_data = positions_data,
			title = f'Reconstruction accuracy with {reconstructor_name}<br><sup>{bureaucrat.run_name}</sup>',
			aspect = 'equal',
			origin = 'lower',
			zmin = 0,
			zmax = 33e-6 if 'nanstd' in col else 33e-6 if 'nanmean' in col else None,
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'reconstruction_accuracy_heatmap.html',
			include_plotlyjs = 'cdn',
		)
		fig = utils.plot_as_xy_contour(
			z = reconstruction_error,
			positions_data = positions_data,
		)
		fig.update_layout(
			title = f'Reconstruction accuracy with {reconstructor_name}<br><sup>{bureaucrat.run_name}</sup>',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/f'reconstruction_accuracy_contour.html',
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
	parser.add_argument('--reconstructor',
		metavar = 'path', 
		help = 'Path to a `reconstructor.pickle`.',
		required = True,
		dest = 'path_to_reconstructor',
		type = str,
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	reconstruct(
		bureaucrat = bureaucrat,
		path_to_reconstructor_pickle = Path(args.path_to_reconstructor),
	)

