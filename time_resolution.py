from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import numpy
import utils
from reconstructors import OnePadTimeReconstructor, MultipadWeightedTimeReconstructor

def time_resolution_analysis(bureaucrat:RunBureaucrat):
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
	
	time_data = data[[f't_{x} (s)' for x in [10,20,30,40,50,60,70,80,90]]]
	
	time_data = time_data.unstack('n_pulse')
	
	# Because the two laser pulses hit on the exact same detector and place, there should be no need to calculate Delta_t sweeping the two k_cfd, it should be symmetric. There may be a slight asymmetry due to the fact that the second laser pulse is a bit attenuated due to the extra 20 m of optic fiber, but I will neglect that here.
	Delta_t = pandas.DataFrame(index=time_data.index)
	for k_cfd in [10,20,30,40,50,60,70,80,90]:
		Delta_t[f'Δt_{k_cfd} (s)'] = time_data[(f't_{k_cfd} (s)',2)] - time_data[(f't_{k_cfd} (s)',1)]
	
	pad_activation = data.query('n_pulse==1')['Amplitude (V)']
	pad_activation.reset_index('n_pulse', inplace=True, drop=True)
	pad_activation = pad_activation.to_frame()
	pad_activation = pad_activation.unstack('n_channel')
	_ = pad_activation.max(axis=1)
	for col in pad_activation.columns:
		pad_activation[col] = pad_activation[col] == _
	pad_activation = pad_activation.stack('n_channel')['Amplitude (V)']
	pad_activation.name = 'pad_is_active'
	
	time_resolution = Delta_t.loc[pad_activation]
	time_resolution = time_resolution.groupby('n_position').agg(numpy.nanstd)
	time_resolution.rename(columns={f'Δt_{i} (s)': f'σ_{i} (s)' for i in [10,20,30,40,50,60,70,80,90]}, inplace=True)
	time_resolution.mask((time_resolution>111e-12), inplace=True) # Remove all "large values" so they don't make the plots to look ugly.
	
	with bureaucrat.handle_task('time_resolution') as employee:
		for col in time_resolution.columns:
			try:
				k_cfd = int(col.replace('σ_','').replace(' (s)',''))
				fig = utils.plot_as_xy_heatmap(
					z = time_resolution[col],
					positions_data = positions_data,
					zmin = 0,
					zmax = 111e-12,
					title = f'Time resolution k_cfd={k_cfd}<br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'time_resolution_vs_position_n_channel_k_cfd_{k_cfd}_heatmap.html',
					include_plotlyjs = 'cdn',
				)
				fig = utils.plot_as_xy_contour(
					z = time_resolution[col],
					positions_data = positions_data,
					zmin = 0,
					zmax = 66e-12,
					title = f'Time resolution k_cfd={k_cfd}<br><sup>{bureaucrat.run_name}</sup>',
					smoothing_sigma = 2,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'time_resolution_vs_position_n_channel_k_cfd_{k_cfd}_contour.html',
					include_plotlyjs = 'cdn',
				)
			except ValueError as e:
				if 'Length mismatch:' in repr(e):
					continue
				else:
					raise e
		for n_channel,df in Delta_t.groupby('n_channel'):
			df = df.groupby('n_position').agg(numpy.nanstd)
			df = df.min(axis=1)
			df.name = 'Best time resolution (s)'
			fig = utils.plot_as_xy_heatmap(
				z = df,
				positions_data = positions_data,
				zmin = 0,
				zmax = 111e-12,
				title = f'Time resolution<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'time_resolution_vs_position_n_channel_{n_channel}_heatmap.html',
				include_plotlyjs = 'cdn',
			)
			fig = utils.plot_as_xy_contour(
				z = df,
				positions_data = positions_data,
				zmin = 0,
				zmax = 66e-12,
				title = f'Time resolution n_channel {n_channel}<br><sup>{bureaucrat.run_name}</sup>',
				smoothing_sigma = 2,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'time_resolution_vs_position_n_channel_{n_channel}_contour.html',
				include_plotlyjs = 'cdn',
			)
	a

def time_reconstructors_testing(bureaucrat:RunBureaucrat):
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
	
	time_data = data[[f't_{x} (s)' for x in [10,20,30,40,50,60,70,80,90]]]
	
	reconstructor = MultipadWeightedTimeReconstructor()
	features = pandas.DataFrame(index=data.index)
	features['time'] = time_data['t_50 (s)']
	features['weight'] = data['Amplitude (V)']
	features = features.unstack('n_channel')
	reconstructed_time = reconstructor.reconstruct(features=features)
	reconstructed_time.name = 'Reconstructed time (s)'
	reconstructed_time = reconstructed_time.to_frame()
	reconstructed_time = reconstructed_time.unstack('n_pulse')
	Delta_t = reconstructed_time[('Reconstructed time (s)',2)] - reconstructed_time[('Reconstructed time (s)',1)]
	Delta_t.name = 'Δt (s)'
	
	print(Delta_t)

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
	time_reconstructors_testing(bureaucrat)
