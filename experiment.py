from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from grafica.plotly_utils.utils import set_my_template_as_default
import numpy

def plot_heatmaps(df, col):
	figs = {}
	for stat in df[col].columns.get_level_values(0).drop_duplicates():
		numpy_array = numpy.array([df[(col,stat)].query(f'n_channel=={n_channel} and n_pulse==1').to_numpy() for n_channel in df.index.get_level_values('n_channel').drop_duplicates()])
		fig = px.imshow(
			numpy_array,
			title = f'{col} {stat} as a function of position<br><sup>{bureaucrat.run_name}</sup>',
			aspect = 'equal',
			labels = dict(
				color = f'{col} {stat}',
				x = 'n_x',
				y = 'n_y',
			),
			x = df[(col,stat)].columns.get_level_values(0).drop_duplicates(),
			y = df[(col,stat)].index.get_level_values(0).drop_duplicates(),
			facet_col = 0,
			origin = 'lower',
		)
		fig.update_coloraxes(colorbar_title_side='right')
		figs[stat] = fig
	return figs

def calculate_thing(data_values, data_fluctuations, positions_data):
	_ = set(data_values.index.names).union(set(data_values.columns.names))
	if _ != {'n_x','n_y'}:
		raise ValueError(f'Index and columns of `data_values` must be "n_x" and "n_y", but they are {_}')
	
	gradient = numpy.gradient(
		data_values.to_numpy(),
		sorted(set(positions_data['x (m)'])),
		sorted(set(positions_data['y (m)'])),
	)
	gradient = numpy.array(gradient)
	gradient = numpy.sum(gradient**2, axis=0)**.5
	gradient = pandas.DataFrame(
		gradient,
		index = data_values.index,
		columns = data_values.columns,
	)
	return data_fluctuations/gradient

def experiment(bureaucrat:RunBureaucrat):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	with bureaucrat.handle_task('experiment') as employee:
		if len(bureaucrat.list_subruns_of_task('TCT_2D_scan')) != 1:
			raise RuntimeError(f'Run {repr(bureaucrat.run_name)} located in "{bureaucrat.path_to_run_directory}" seems to be corrupted because I was expecting only a single subrun for the task "TCT_2D_scan" but it actually has {len(bureaucrat.list_subruns_of_task("TCT_2D_scan"))} subruns...')
		flattened_1D_scan_subrun_bureaucrat = bureaucrat.list_subruns_of_task('TCT_2D_scan')[0]
		parsed_from_waveforms = load_whole_dataframe(flattened_1D_scan_subrun_bureaucrat.path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite')
		
		positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
		positions_data.reset_index(['n_x','n_y'], drop=False, inplace=True)
		for _ in {'x','y'}: # Remove offset so (0,0) is the center...
			positions_data[f'{_} (m)'] -= positions_data[f'{_} (m)'].mean()
		
		data = parsed_from_waveforms.merge(positions_data, left_index=True, right_index=True)
		data.set_index(['n_x','n_y'], append=True, inplace=True)
		data.reset_index('n_waveform', drop=True, inplace=True)
		
		# ~ data = data.query('`t_50 (s)`>2e-9 and `t_50 (s)`<4e-9')
		
		# Calculate some event-wise stuff...
		data = data.unstack('n_channel')
		for n_channel in data.columns.get_level_values('n_channel').drop_duplicates():
			data[('Time from CH1 (s)',n_channel)] = data[('t_20 (s)',n_channel)] - data[('t_50 (s)',1)]
			data[('Total collected charge (V s)',n_channel)] = data[[('Collected charge (V s)',_) for _ in data.columns.get_level_values('n_channel').drop_duplicates()]].sum(axis=1)
			data[('Charge shared fraction',n_channel)] = data[('Collected charge (V s)',n_channel)]/data[('Total collected charge (V s)',n_channel)]
		data = data.stack('n_channel')
		
		averages_2D = data.groupby(['n_pulse','n_channel','n_x','n_y']).agg([('average',numpy.nanmedian),('fluctuations',numpy.nanstd)])
		averages_2D = pandas.pivot_table(
			data = averages_2D,
			values = averages_2D.columns,
			index = ['n_x','n_channel','n_pulse'],
			columns = 'n_y',
		)
		
		thing = []
		for col in {'Charge shared fraction','Time from CH1 (s)','t_50 (s)','Amplitude (V)','Total collected charge (V s)','Charge shared fraction'}:
			figs = plot_heatmaps(averages_2D, col)
			for stat,fig in figs.items():
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{col} {stat}.html',
					include_plotlyjs = 'cdn',
				)
			for _,df in averages_2D.groupby(['n_channel','n_pulse']):
				n_channel, n_pulse = _
				df = df.reset_index(['n_channel','n_pulse'], drop=True)
				_ = calculate_thing(df[(col,'average')], df[(col,'fluctuations')], positions_data)
				thing.append(
					{
						'n_channel': n_channel,
						'n_pulse': n_pulse,
						'thing': _,
						'variable': f"Thing({col}) (m)",
					}
				)
		thing = pandas.concat(
			[_['thing'] for _ in thing],
			keys = [(_['variable'],_['n_channel'],_['n_pulse']) for _ in thing],
			axis = 0,
		)
		thing.index.set_names(['variable','n_channel','n_pulse'], level=[0,1,2], inplace=True)
		
		for variable, df in thing.groupby('variable'):
			df = df.query('n_pulse==1')
			df.reset_index(['variable','n_pulse'], drop=True, inplace=True)
			numpy_array = numpy.array([_df.to_numpy() for n_channel,_df in df.groupby('n_channel')])
			fig = px.imshow(
				numpy_array,
				title = f'{variable}<br><sup>{bureaucrat.run_name}</sup>',
				aspect = 'equal',
				labels = dict(
					color = variable,
					x = 'n_x',
					y = 'n_y',
				),
				x = sorted(set(positions_data['x (m)'])),
				y = sorted(set(positions_data['y (m)'])),
				zmin = 0,
				zmax = 33e-6,
				facet_col = 0,
				origin = 'lower',
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{variable}.html',
				include_plotlyjs = 'cdn',
			)
			
		
		# Calculate variables...ValueError: Index contains duplicate entries, cannot reshape

		data = data.unstack('n_channel')
		variables = {}
		variables['f_horizontal'] = data[('Charge shared fraction',1)] + data[('Charge shared fraction',3)] - data[('Charge shared fraction',2)] - data[('Charge shared fraction',4)]
		variables['f_vertical'] = data[('Charge shared fraction',1)] + data[('Charge shared fraction',2)] - data[('Charge shared fraction',3)] - data[('Charge shared fraction',4)]
		variables['log_14'] = numpy.log10(data[('Collected charge (V s)',1)]/data[('Collected charge (V s)',4)])
		variables['log_23'] = numpy.log10(data[('Collected charge (V s)',2)]/data[('Collected charge (V s)',3)])
		for n_channel in [2,3,4]:
			variables[f'Time from CH1 of CH{n_channel} (s)'] = data[('Time from CH1 (s)',n_channel)]
		for _,_2 in variables.items():
			_2.name = _
		data = data.stack('n_channel')
		variables = pandas.concat([item for _,item in variables.items()], axis=1)
		
		variables = variables.groupby(['n_pulse','n_x','n_y']).agg([('average',numpy.nanmedian),('fluctuations',numpy.nanstd)])
		for col in variables.columns.get_level_values(0).drop_duplicates():
			variables[(col,'relative fluctuations')] = variables[(col,'fluctuations')] / abs(variables[(col,'average')])
		variables = pandas.pivot_table(
			data = variables,
			values = variables.columns,
			index = ['n_x','n_pulse'],
			columns = 'n_y',
		)
		
		for col in variables.columns.get_level_values(0).drop_duplicates():
			for stat in variables[col].columns.get_level_values(0).drop_duplicates():
				df = variables.query('n_pulse==1').reset_index('n_pulse', drop=True)[(col,stat)]
				fig = px.imshow(
					df,
					title = f'{col} {stat}<br><sup>{bureaucrat.run_name}</sup>',
					zmax = .4,#numpy.nanmedian(df) + 2*numpy.nanstd(df),
					zmin = 0,#numpy.nanmedian(df) - 2*numpy.nanstd(df),
					origin = 'lower',
				)
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{col}_{stat}.html',
					include_plotlyjs = 'cdn',
				)

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
	experiment(bureaucrat)

