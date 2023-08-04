from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import utils
import numpy
import logging
import plotly.express as px
import json
import plotly_utils

def compare_position_reconstrucitons(bureaucrat:RunBureaucrat):
	NAMES_BEGIN_WITH = 'reconstruction_using_position_reconstructor_'
	NAMES_END_WITH = '_analysis'
	paths_to_tasks_resulting_from_a_position_reconstruction_analysis = [p for p in bureaucrat.path_to_run_directory.iterdir() if p.name[:len(NAMES_BEGIN_WITH)]==NAMES_BEGIN_WITH and p.name[-len(NAMES_END_WITH):]==NAMES_END_WITH]
	
	reconstruction_errors = []
	reconstructors_info = []
	for p in paths_to_tasks_resulting_from_a_position_reconstruction_analysis:
		bureaucrat.check_these_tasks_were_run_successfully([p.name, p.name.replace('_analysis','')])
		with open(bureaucrat.path_to_directory_of_task(p.name.replace('_analysis',''))/'reconstructor_info.json', 'r') as ifile:
			reconstructor_info = json.load(ifile)
		_reconstruction_error = pandas.read_pickle(bureaucrat.path_to_directory_of_task(p.name)/'reconstruction_error.pickle').to_frame()
		_reconstruction_error['reconstructor_name'] = reconstructor_info['reconstructor_name']
		_reconstruction_error.set_index('reconstructor_name', append=True, inplace=True)
		_reconstruction_error = _reconstruction_error.reorder_levels(['reconstructor_name'] + list(_reconstruction_error.index.names)[:-1])
		_reconstruction_error = _reconstruction_error['Reconstruction error (m)']
		reconstruction_errors.append(_reconstruction_error)
		
		_reconstructor_info = dict(
			reconstructor_name = reconstructor_info['reconstructor_name'],
			reconstructor_type = reconstructor_info['reconstructor_name'].split('_')[2].split('PositionReconstructor')[0],
			reconstructor_x_grid = int(reconstructor_info['reconstructor_name'].split('_')[-1].split('x')[0]),
			reconstructor_y_grid = int(reconstructor_info['reconstructor_name'].split('_')[-1].split('x')[1]),
		)
		reconstructors_info.append(_reconstructor_info)
	reconstruction_errors = pandas.concat(reconstruction_errors)
	reconstructors_info = pandas.DataFrame.from_records(reconstructors_info).set_index('reconstructor_name')
	
	def q99(x):
		return numpy.quantile(x, .99)
	statistics = reconstruction_errors.groupby('reconstructor_name').agg([numpy.nanmean,numpy.nanstd,q99])
	statistics.rename(
		columns = {
			'nanstd': 'Reconstruction error std (m)',
			'nanmean': 'Reconstruction error mean (m)',
			'q99': 'Reconstruction error q99 (m)',
		},
		inplace = True,
	)
	
	with bureaucrat.handle_task('compare_position_reconstrucitons') as employee:
		
		DUT_PITCH = 500e-6
		LABELS_FOR_PLOTS = {
			'reconstructor_type': 'Reconstructor',
		}
		fig = px.ecdf(
			reconstruction_errors.to_frame().sample(n=5555).merge(reconstructors_info,left_index=True, right_index=True).reset_index(drop=False).sort_values(['reconstructor_type','reconstructor_x_grid']),
			x = 'Reconstruction error (m)',
			line_dash = 'reconstructor_type',
			color = 'reconstructor_x_grid',
			title = f'Reconstruction error distribution<br><sup>{bureaucrat.run_name}</sup>',
			labels = LABELS_FOR_PLOTS,
		)
		fig.add_vline(
				x = DUT_PITCH*(2/12)**.5,
				annotation_text = f'Binary readout uncertainty = {DUT_PITCH*(2/12)**.5*1e6:.0f} µm',
				line_dash = 'dash',
				annotation_textangle = -90,
			)
		fig.write_html(
			employee.path_to_directory_of_my_task/'reconstruction_error_ecdf.html',
			include_plotlyjs = 'cdn',
		)
		
		for col in statistics:
			fig = px.line(
				statistics.merge(reconstructors_info,left_index=True, right_index=True).reset_index(drop=False).sort_values(['reconstructor_type','reconstructor_x_grid']),
				title = f'{col.replace(" (m)","")}<br><sup>{bureaucrat.run_name}</sup>',
				x = 'reconstructor_x_grid',
				y = col,
				color = 'reconstructor_type',
				markers = True,
				labels = LABELS_FOR_PLOTS,
			)
			fig.add_hline(
				y = DUT_PITCH*(2/12)**.5,
				annotation_text = f'Binary readout uncertainty = {DUT_PITCH*(2/12)**.5*1e6:.0f} µm',
				line_dash = 'dash',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}.html',
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
	compare_position_reconstrucitons(bureaucrat)
