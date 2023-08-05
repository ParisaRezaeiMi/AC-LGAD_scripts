from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import utils
import numpy
import logging
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly_utils

def sample_reconstruction_error_for_square_binary_pixel(pitch, n:int):
	x = numpy.random.rand(n)*pitch-pitch/2
	y = numpy.random.rand(n)*pitch-pitch/2
	reconstruction_error = (x**2+y**2)**.5
	return reconstruction_error

def ecdf(x):
	xs = numpy.sort(x)
	ys = numpy.arange(1, len(xs)+1)/float(len(xs))
	return xs, ys

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
			reconstructor_x_grid_n_points = int(reconstructor_info['reconstructor_name'].split('_')[-1].split('x')[0]),
			reconstructor_y_grid_n_points = int(reconstructor_info['reconstructor_name'].split('_')[-1].split('x')[1]),
		)
		reconstructors_info.append(_reconstructor_info)
	reconstruction_errors = pandas.concat(reconstruction_errors)
	reconstructors_info = pandas.DataFrame.from_records(reconstructors_info).set_index('reconstructor_name')
	
	positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
	positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
	for _ in {'x','y'}: # Remove offset so (0,0) is the center...
		positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
	
	for xy in {'x','y'}:
		reconstructors_info[f'Reconstructor {xy} pitch (m)'] = (positions_data[f'{xy} (m)'].max() - positions_data[f'{xy} (m)'].min())/reconstructors_info[f'reconstructor_{xy}_grid_n_points']
	
	def q99(x):
		return numpy.quantile(x, .99)
	def q999(x):
		return numpy.quantile(x, .999)
	def q100(x):
		return numpy.quantile(x, 1)
	statistics = reconstruction_errors.groupby('reconstructor_name').agg([numpy.nanmean,numpy.nanstd,q99,q100,q999])
	statistics.rename(
		columns = {
			'nanstd': 'Reconstruction error std (m)',
			'nanmean': 'Reconstruction error mean (m)',
			'q99': 'Reconstruction error q99 (m)',
			'q100': 'Reconstruction error q100 (m)',
			'q999': 'Reconstruction error q999 (m)',
		},
		inplace = True,
	)
	
	with bureaucrat.handle_task('compare_position_reconstrucitons') as employee:
		
		DUT_PITCH = 500e-6
		LABELS_FOR_PLOTS = {
			'reconstructor_type': 'Reconstructor',
			'Reconstruction error q99 (m)': 'Reconstruction error q<sub>99%</sub> (m)',
			'Reconstruction error q999 (m)': 'Reconstruction error q<sub>99.9%</sub> (m)',
			'Reconstruction error q100 (m)': 'Reconstruction error q<sub>100%</sub> (m)',
			'reconstructor_x_grid_n_points': 'Reconstructor grid N×N',
		}
		
		# ECDF plot ----------------------------------------------------
		data_for_ECDF_plot = reconstruction_errors.to_frame().merge(reconstructors_info,left_index=True, right_index=True).query('reconstructor_x_grid_n_points in [2,3,8,18]').sample(n=5555).reset_index(drop=False).sort_values(['reconstructor_type','reconstructor_x_grid_n_points'])
		fig = px.ecdf(
			data_for_ECDF_plot,
			x = 'Reconstruction error (m)',
			facet_row = 'reconstructor_type',
			color = 'reconstructor_x_grid_n_points',
			title = f'Reconstruction error distribution<br><sup>{bureaucrat.run_name}</sup>',
			labels = LABELS_FOR_PLOTS,
		)
		for row in [1,2,3]:
			for dut_pitch in [DUT_PITCH] + list(data_for_ECDF_plot['Reconstructor x pitch (m)'].drop_duplicates()):
				x,y = ecdf(sample_reconstruction_error_for_square_binary_pixel(pitch=dut_pitch, n=99999))
				_ = pandas.DataFrame({'x':x,'y':y})
				_ = _.iloc[numpy.arange(0,len(_)-1,int(len(_)/99))]
				_name = f'{dut_pitch*1e6:.0f}×{dut_pitch*1e6:.0f} µm<sup>2</sup> binary readout pixel'
				fig.add_trace(
					go.Scatter(
						x = _['x'],
						y = _['y'],
						# ~ line_shape = 'hv',
						name = _name,
						line_color = 'black',
						showlegend = True if row==1 else False,
						legendgroup = _name,
					),
					row = row,
					col = 1,
				)
		fig.write_html(
			employee.path_to_directory_of_my_task/'reconstruction_error_ecdf.html',
			include_plotlyjs = 'cdn',
		)
		
		for col in statistics:
			fig = px.line(
				statistics.merge(reconstructors_info,left_index=True, right_index=True).reset_index(drop=False).sort_values(['reconstructor_type','reconstructor_x_grid_n_points']),
				title = f'{col.replace(" (m)","")}<br><sup>{bureaucrat.run_name}</sup>',
				x = 'reconstructor_x_grid_n_points',
				y = col,
				color = 'reconstructor_type',
				markers = True,
				labels = LABELS_FOR_PLOTS,
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}.html',
				include_plotlyjs = 'cdn',
			)
		
		quantiles = reconstruction_errors.groupby('reconstructor_name').quantile([.5,.95,.99])
		quantiles = quantiles.to_frame()
		quantiles.reset_index(level=-1,drop=False,inplace=True)
		quantiles.columns = ['quantile (%)'] + list(quantiles.columns[1:])
		quantiles['quantile (%)'] *= 100
		for x_axis_variable_name in {'Reconstructor x pitch (m)','reconstructor_x_grid_n_points'}:
			fig = px.line(
				quantiles.merge(reconstructors_info,left_index=True, right_index=True).reset_index(drop=False).sort_values(['reconstructor_type',x_axis_variable_name,'quantile (%)']),
				title = f'Reconstruction algorithms comparison<br><sup>{bureaucrat.run_name}</sup>',
				x = x_axis_variable_name,
				y = 'Reconstruction error (m)',
				color = 'reconstructor_type',
				markers = True,
				labels = LABELS_FOR_PLOTS,
				facet_col = 'quantile (%)',
				log_y = True,
			)
			for col,q in enumerate(sorted(quantiles['quantile (%)'].drop_duplicates())):
				error_q = numpy.quantile(a=sample_reconstruction_error_for_square_binary_pixel(DUT_PITCH,n=99999), q=q/100)
				fig.add_trace(
					go.Scatter(
						x = [numpy.mean([reconstructors_info[x_axis_variable_name].min(),reconstructors_info[x_axis_variable_name].max()])],
						y = [error_q],
						text=[f'{DUT_PITCH*1e6:.0f}×{DUT_PITCH*1e6:.0f} µm<sup>2</sup> binary readout pixel<br>'],
						mode="text",
						showlegend = False,
					),
					row = 1,
					col = col+1,
				)
				fig.add_shape(
					go.layout.Shape(
						type = "line",
						yref = "y",
						xref = "x domain",
						x0 = 0,
						y0 = error_q,
						x1 = 1,
						y1 = error_q,
						line = dict(dash='dash'),
					),
					row = 1,
					col = col+1,
				)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'reconstructors_comparison_{x_axis_variable_name}.html',
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
