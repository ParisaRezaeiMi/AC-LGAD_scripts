from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import pickle
import reconstructors
import utils
import numpy
import logging
import sqlite3
import json
from train_reconstructors import calculate_features, features_parisa
import matplotlib.pyplot as plt
import plotly.express as px

POSITION_VARIABLES_NAMES = ['x (m)','y (m)']


# ~ def reconstruct(bureaucrat:RunBureaucrat, path_to_reconstructor_pickle:Path, force:bool=False):
	# ~ bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	# ~ reconstructor_bureaucrat = RunBureaucrat(path_to_reconstructor_pickle.parent.parent)
	# ~ reconstructor_name = path_to_reconstructor_pickle.parent.parts[-1]
	# ~ reconstructor_bureaucrat.check_these_tasks_were_run_successfully(reconstructor_name)
	
	# ~ this_task_name = f'reconstruction_using_{reconstructor_name}'
	
	# ~ if force==False and bureaucrat.was_task_run_successfully(this_task_name):
		# ~ logging.info(f'Task {repr(this_task_name)} was already performed successfully, skipping reconstruction.')
		# ~ return
	
	# ~ with open(path_to_reconstructor_pickle, 'rb') as ifile:
		# ~ reconstructor = pickle.load(ifile)
	
	# ~ logging.info(f'Loading data from {bureaucrat.run_name}...')
	# ~ if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
		# ~ raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
	# ~ flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
	# ~ connection = sqlite3.connect(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
	# ~ data = pandas.read_sql("SELECT n_position,n_trigger,n_channel,`Amplitude (V)` FROM dataframe_table WHERE n_pulse==1", connection)
	# ~ data.set_index(['n_position','n_trigger','n_channel'], inplace=True)
	
	# ~ positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	# ~ positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	# ~ for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		# ~ positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	
	# ~ # Calculate some event-wise stuff:
	# ~ data = data.unstack('n_channel')
	# ~ for n_channel in data.columns.get_level_values('n_channel').drop_duplicates():
		# ~ data[('Total amplitude (V)',n_channel)] = data[[('Amplitude (V)',_) for _ in data.columns.get_level_values('n_channel').drop_duplicates()]].sum(axis=1)
		# ~ data[('ASF',n_channel)] = data[('Amplitude (V)',n_channel)]/data[('Total amplitude (V)',n_channel)]
	# ~ data = data.stack('n_channel') # Revert what I have done before.
	
	# ~ amplitude_shared_fraction = data[['ASF']].unstack('n_channel')
	# ~ amplitude_shared_fraction.columns = [f'ASF {n_channel}' for n_channel in amplitude_shared_fraction.columns.get_level_values('n_channel')]
	
	# ~ with bureaucrat.handle_task(this_task_name) as employee:
		# ~ json.dump(
			# ~ dict(
				# ~ reconstructor_name = reconstructor_name,
				# ~ path_to_reconstructor_pickle = str(path_to_reconstructor_pickle.resolve()),
			# ~ ),
			# ~ open(employee.path_to_directory_of_my_task/'reconstructor_info.json','w')
		# ~ )
		
		# ~ testing_data = calculate_features(data)
		
		# ~ path_for_plots = employee.path_to_directory_of_my_task/'features'
		# ~ for col in reconstructor.features_names:
			# ~ fig = utils.plot_as_xy_heatmap(
				# ~ z = testing_data.groupby('n_position').agg(numpy.nanmean)[col],
				# ~ positions_data = positions_data,
				# ~ title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
				# ~ aspect = 'equal',
				# ~ origin = 'lower',
			# ~ )
			# ~ path_for_plots.mkdir(exist_ok=True)
			# ~ fig.write_html(
				# ~ path_for_plots/f'{col}_heatmap.html',
				# ~ include_plotlyjs = 'cdn',
			# ~ )
			# ~ fig = utils.plot_as_xy_contour(
				# ~ z = testing_data.groupby('n_position').agg(numpy.nanmean)[col],
				# ~ positions_data = positions_data,
				# ~ smoothing_sigma = 2,
			# ~ )
			# ~ fig.update_layout(
				# ~ title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
			# ~ )
			# ~ fig.write_html(
				# ~ path_for_plots/f'{col}_contour.html',
				# ~ include_plotlyjs = 'cdn',
			# ~ )
		
		# ~ logging.info(f'Reconstructing...')
		# ~ reconstructed = reconstructor.reconstruct(
			# ~ testing_data[reconstructor.features_names],
			# ~ batch_size = 11111,
		# ~ )
		# ~ utils.save_dataframe(
			# ~ reconstructed,
			# ~ name = 'reconstructed_positions',
			# ~ location = employee.path_to_directory_of_my_task,
			# ~ formats = ['pickle','feather'],
		# ~ )
		# ~ logging.info(f'Finished reconstruction of {bureaucrat.run_name}.')







def reconstruct(bureaucrat:RunBureaucrat, path_to_reconstructor_pickle:Path, force:bool=False):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	reconstructor_bureaucrat = RunBureaucrat(path_to_reconstructor_pickle.parent.parent)
	reconstructor_name = path_to_reconstructor_pickle.parent.parts[-1]
	reconstructor_bureaucrat.check_these_tasks_were_run_successfully(reconstructor_name)
	
	this_task_name = f'reconstruction_using_{reconstructor_name}'
	
	if force==False and bureaucrat.was_task_run_successfully(this_task_name):
		logging.info(f'Task {repr(this_task_name)} was already performed successfully, skipping reconstruction.')
		return
	
	with open(path_to_reconstructor_pickle, 'rb') as ifile:
		reconstructor = pickle.load(ifile)
	
	logging.info(f'Loading data from {bureaucrat.run_name}...')
	if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
		raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
	flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
	connection = sqlite3.connect(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
	data = pandas.read_sql("SELECT n_position,n_trigger,n_channel,`t_50 (s)` ,`Amplitude (V)`  FROM dataframe_table WHERE n_pulse==1", connection)
	data.set_index(['n_position','n_trigger','n_channel'], inplace=True)
	
	positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	
	
	with bureaucrat.handle_task(this_task_name) as employee:
		json.dump(
			dict(
				reconstructor_name = reconstructor_name,
				path_to_reconstructor_pickle = str(path_to_reconstructor_pickle.resolve()),
			),
			open(employee.path_to_directory_of_my_task/'reconstructor_info.json','w')
		)
		
#		testing_data = features(data) #HERE is my change
		testing_data = features_parisa(data)
		#print(testing_data)
		#a
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
				smoothing_sigma = 2,
			)
			fig.update_layout(
				title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				path_for_plots/f'{col}_contour.html',
				include_plotlyjs = 'cdn',
			)
		logging.info(f'Reconstructing...')
		reconstructed = reconstructor.reconstruct(
			testing_data[reconstructor.features_names],
			batch_size = 11111,
		)
		utils.save_dataframe(
			reconstructed,
			name = 'reconstructed_positions',
			location = employee.path_to_directory_of_my_task,
			formats = ['pickle','feather'],
		)
		logging.info(f'Finished reconstruction of {bureaucrat.run_name}.')





def analyze_reconstruction(bureaucrat:RunBureaucrat, reconstruction_task_name:str):
	bureaucrat.check_these_tasks_were_run_successfully(reconstruction_task_name)
	
	logging.info(f'Loading data from {bureaucrat.run_name}...')
	if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
		raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
	flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
	connection = sqlite3.connect(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
	data = pandas.read_sql("SELECT n_position,n_trigger FROM dataframe_table WHERE n_pulse==1 AND n_channel==1", connection)
	data.set_index(['n_position','n_trigger'], inplace=True)
	data = data.drop_duplicates()
	
	positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	
	original = data.merge(positions_data[POSITION_VARIABLES_NAMES], left_index=True, right_index=True)
	
	reconstructed = pandas.read_pickle(bureaucrat.path_to_directory_of_task(reconstruction_task_name)/'reconstructed_positions.pickle')
	
	logging.info('Processing data...')
	
	print(original)
	print(reconstructed)
	
	reconstruction_error = numpy.sum((original-reconstructed)**2, axis=1)**.5
	reconstruction_error.name = 'Reconstruction error (m)'
	
	def q99(x):
		return numpy.quantile(x, .9)
	result = reconstruction_error.groupby('n_position').agg([numpy.nanmean,numpy.nanstd,q99])
	result.rename(
		columns = {
			'nanstd': 'Reconstruction error std (m)',
			'nanmean': 'Reconstruction error mean (m)',
			'q99': 'Reconstruction error q99 (m)',
		},
		inplace = True,
	)
	
	with open(bureaucrat.path_to_directory_of_task(reconstruction_task_name)/'reconstructor_info.json', 'r') as ifile:
		reconstructor_info = json.load(ifile)
	
	with bureaucrat.handle_task(f'{reconstruction_task_name}_analysis') as employee:
		logging.info('Saving reconstruction error...')
		utils.save_dataframe(
			df = reconstruction_error,
			name = 'reconstruction_error',
			location = employee.path_to_directory_of_my_task,
			formats = ['feather','pickle'],
		)
		logging.info('Producing and saving plots...')
		subtitle_lines = [f'Reconstructor: {reconstructor_info["reconstructor_name"]}',f'Training data: {Path(reconstructor_info["path_to_reconstructor_pickle"]).parent.parent.name}']
		for col in result.columns:
			title = f'{col.replace(" (m)","")}<br><sup>{bureaucrat.run_name}</sup>'
			title += '<br><sup>' + '</sup><br><sup>'.join(subtitle_lines) + '</sup>'
			fig = utils.plot_as_xy_heatmap(
				z = result[col],
				positions_data = positions_data,
				title = title,
				aspect = 'equal',
				origin = 'lower',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_heatmap.html',
				include_plotlyjs = 'cdn',
			)
			fig = utils.plot_as_xy_contour(
				z = result[col],
				positions_data = positions_data,
				smoothing_sigma = 2,
			)
			fig.update_layout(
				title = title,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_contour.html',
				include_plotlyjs = 'cdn',
			)
			
		# Quiver plot with biases ---
		reconstructed_average = reconstructed.groupby('n_position').agg(numpy.nanmean)
		reconstructed_average.rename(columns={'x (m)': 'x (m) reconstructed', 'y (m)': 'y (m) reconstructed'}, inplace=True)
		z = reconstructed_average.merge(positions_data[['n_x','n_y','x (m)','y (m)']], left_index=True, right_index=True)
		z.set_index(['n_x','n_y'], inplace=True)
		z = z.unstack('n_x')
		xx,yy = numpy.meshgrid(sorted(set(positions_data['x (m)'])), sorted(set(positions_data['y (m)'])))
		fig, ax = plt.subplots()
		ax.quiver(
			xx*1e6,
			yy*1e6,
			z['x (m) reconstructed'] - xx,
			z['y (m) reconstructed'] - yy,
			angles = 'xy', 
			scale_units = 'xy', 
			scale = 1e-6,
			edgecolor = 'blue',
		)
		ax.scatter(
			x = positions_data['x (m)']*1e6,
			y = positions_data['y (m)']*1e6,
			c = 'red',
			s = 1,
		)
		ax.set_aspect('equal')
		ax.set_xlabel('x (µm)')
		ax.set_ylabel('y (µm)')
		plt.title(f'Reconstruction bias plot\n{bureaucrat.run_name}')
		for fmt in {'pdf'}:
			plt.savefig(employee.path_to_directory_of_my_task/f'reconstruction_error_average_quiver.{fmt}')
		
		fig = px.ecdf(
			title = f'Reconstruction error distribution<br><sup>{bureaucrat.run_name}</sup>' + '<br><sup>' + '</sup><br><sup>'.join(subtitle_lines) + '</sup>',
			data_frame = reconstruction_error.to_frame().sample(n=5555).reset_index(drop=False),
			x = 'Reconstruction error (m)',
			marginal = 'histogram',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'reconstruction_error_distribution.html',
			include_plotlyjs = 'cdn',
		)
		
		logging.info(f'Finished with {bureaucrat.run_name}.')

def reconstruct_and_analyze(bureaucrat:RunBureaucrat, path_to_reconstructor_pickle:Path, force:bool=False):
	reconstruct(
		bureaucrat = bureaucrat,
		path_to_reconstructor_pickle = path_to_reconstructor_pickle,
		force = force,
	)
	analyze_reconstruction(
		bureaucrat = bureaucrat,
		reconstruction_task_name = f'reconstruction_using_{path_to_reconstructor_pickle.parent.name}',
	)

def reconstruct_and_analyze_using_all_available_reconstructors(bureaucrat:RunBureaucrat, bureaucrat_where_to_search_for_reconstructors:RunBureaucrat, force:bool=False):
	possible_tasks_having_to_do_with_trained_reconstructors = [p.name for p in bureaucrat_where_to_search_for_reconstructors.path_to_run_directory.iterdir() if p.name[:len('position_reconstructor_')]=='position_reconstructor_']
	
	for task_name in possible_tasks_having_to_do_with_trained_reconstructors:
		if not bureaucrat_where_to_search_for_reconstructors.was_task_run_successfully(task_name):
			continue
		logging.info(f'Found task called {task_name} in {bureaucrat_where_to_search_for_reconstructors.run_name} that looks like a reconstructor, will attempt to use it...')
		reconstruct_and_analyze(
			bureaucrat = bureaucrat,
			path_to_reconstructor_pickle = bureaucrat_where_to_search_for_reconstructors.path_to_directory_of_task(task_name)/'reconstructor.pickle',
			force = force,
		)
		logging.info(f'Finished reconstructing with {task_name}.')
	

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
		type = str,
	)
	parser.add_argument('--reconstructor',
		metavar = 'path', 
		help = 'Path to a `reconstructor.pickle`.',
		required = True,
		dest = 'path_to_reconstructor',
		type = str,
	)
	parser.add_argument(
		'--force',
		help = 'If this flag is passed, it will force the calculation even if it was already done beforehand. Old data will be deleted.',
		required = False,
		dest = 'force',
		action = 'store_true'
	)
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	reconstruct_and_analyze(
		bureaucrat = bureaucrat,
		path_to_reconstructor_pickle = Path(args.path_to_reconstructor),
		force = args.force,
	)
	# ~ reconstruct_and_analyze_using_all_available_reconstructors(
		# ~ bureaucrat = bureaucrat,
		# ~ bureaucrat_where_to_search_for_reconstructors = RunBureaucrat(Path(args.path_to_reconstructor)),
		# ~ force = args.force,
	# ~ )
