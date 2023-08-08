from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import numpy
import utils
from reconstructors import OnePadTimeReconstructor, MultipadWeightedTimeReconstructor
import logging

def time_reconstructors_testing(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	POSITION_VARIABLES_NAMES = ['x (m)','y (m)']
	
	# Load data:
	logging.info(f'Loading data from {repr(bureaucrat.run_name)}...')
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
	
	RECONSTRUCTORS_TO_TEST = [
		dict(
			reconstructor = OnePadTimeReconstructor(),
			name = 'OnePadTimeReconstructor',
		),
	] + [
		dict(
			reconstructor = MultipadWeightedTimeReconstructor(n=n),
			name = f'MultipadWeightedTimeReconstructorN{n}',
		) for n in [1,2,3,4,5,6,7,8,9,10]
	]
	
	for reco_stuff in RECONSTRUCTORS_TO_TEST:
		reconstructor_name = reco_stuff["name"]
		logging.info(f'Processing reconstructor {repr(reconstructor_name)}...')
		with bureaucrat.handle_task(f'time_resolution_using_{reconstructor_name.replace(" ","_")}') as employee:
			for k_CFD in [10,20,30,40,50,60,70,80,90]:
				reconstructor = reco_stuff['reconstructor']
				features = pandas.DataFrame(index=data.index)
				features['time'] = data[f't_{k_CFD} (s)']
				features['weight'] = data['Amplitude (V)']
				features = features.unstack('n_channel')
				reconstructed_time = reconstructor.reconstruct(features=features)
				reconstructed_time.name = 'Reconstructed time (s)'
				reconstructed_time = reconstructed_time.to_frame()
				reconstructed_time = reconstructed_time.unstack('n_pulse')
				Delta_t = reconstructed_time[('Reconstructed time (s)',2)] - reconstructed_time[('Reconstructed time (s)',1)]
				Delta_t.name = 'Δt (s)'
				
				time_resolution = Delta_t.groupby('n_position').agg(numpy.nanstd)
				time_resolution.name = 'Time resolution (s)'
				
				fig = utils.plot_as_xy_heatmap(
					z = time_resolution,
					positions_data = positions_data,
					zmin = 0,
					zmax = 111e-12,
					title = f'Time resolution vs position<br><sup>k_CFD={k_CFD}, reconstructor: {reconstructor_name}</sup><br><sup>{bureaucrat.run_name}</sup>',
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'time_resolution_vs_position_k_CFD_{k_CFD}_heatmap.html',
					include_plotlyjs = 'cdn',
				)
				fig = utils.plot_as_xy_contour(
					z = time_resolution,
					positions_data = positions_data,
					zmin = 0,
					zmax = 66e-12,
					title = f'Time resolution vs position<br><sup>k_CFD={k_CFD}, reconstructor: {reconstructor_name}</sup><br><sup>{bureaucrat.run_name}</sup>',
					smoothing_sigma = 2,
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'time_resolution_vs_position_k_CFD_{k_CFD}_contour.html',
					include_plotlyjs = 'cdn',
				)
				

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
	
	args = parser.parse_args()
	bureaucrat = RunBureaucrat(Path(args.directory))
	time_reconstructors_testing(bureaucrat)
