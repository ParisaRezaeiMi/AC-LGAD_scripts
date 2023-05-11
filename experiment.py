from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from grafica.plotly_utils.utils import set_my_template_as_default
import numpy

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
		
		# Calculate time from CH1...
		data['Time from CH1 (s)'] = float('NaN')
		data = data.unstack('n_channel')
		for n_channel in data['Time from CH1 (s)'].columns:
			data[('Time from CH1 (s)',n_channel)] = data[('t_20 (s)',n_channel)] - data[('t_50 (s)',1)]
		data = data.stack('n_channel')
		
		# Calculate charge fraction for each channel...
		total_collected_charge = data.groupby(['n_position','n_trigger','n_pulse','n_x','n_y'])['Collected charge (V s)'].sum()
		total_collected_charge.name = 'Total collected charge (V s)'
		_ = data['Collected charge (V s)']/total_collected_charge
		_.name = 'Charge shared fraction'
		data = data.join(_)
		
		_ = data.groupby(['n_pulse','n_channel','n_x','n_y']).agg(numpy.nanmean)
		averages_2D = pandas.pivot_table(
			data = _,
			values = _.columns,
			index = ['n_x','n_channel','n_pulse'],
			columns = 'n_y',
		)
		
		for col in {'Charge shared fraction','Time from CH1 (s)','t_50 (s)','Amplitude (V)'}:
			fig = px.imshow(
				1e9*numpy.array([averages_2D[col].query(f'n_channel=={n_channel} and n_pulse==1').to_numpy() for n_channel in sorted(set(averages_2D.index.get_level_values('n_channel')))]),
				title = f'{col} as a function of position<br><sup>{bureaucrat.run_name}</sup>',
				aspect = 'equal',
				labels = dict(
					color = col,
					x = 'n_x',
					y = 'n_y',
				),
				x = averages_2D[col].columns,
				y = averages_2D[col].index.get_level_values(0).drop_duplicates(),
				facet_col = 0,
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_imshow.html',
				include_plotlyjs = 'cdn',
			)
			
			fig = make_subplots(
				rows = 1, 
				cols = len(set(data.index.get_level_values('n_channel'))),
				shared_xaxes = True, 
				shared_yaxes = True,
				subplot_titles = ['dummy title' for _ in range(len(set(data.index.get_level_values('n_channel'))))],
			)
			for i,n_channel in enumerate(sorted(set(data.index.get_level_values('n_channel')))):
				fig.add_trace(
					go.Contour(
						z = averages_2D[col].query(f'n_channel=={n_channel}').to_numpy(),
						x = averages_2D[col].columns,
						y = averages_2D[col].index.get_level_values(0).drop_duplicates(),
						zmin = averages_2D[col].min().min(),
						zmax = averages_2D[col].max().max(),
						showscale = i==0,
						contours = dict(
							coloring ='heatmap',
							showlabels = True, # show labels on contours
							labelfont = dict( # label font properties
								size = 12,
								color = 'white',
							),
						),
						line = dict(color = "white"),
					),
					row = 1,
					col = i+1,
				)
				fig.layout.annotations[i].update(text=f'n_channel:{n_channel}')
				fig.update_xaxes(title_text = averages_2D[col].columns.name, row=1, col=i+1)
			fig.update_yaxes(title_text=averages_2D[col].index.names[0], row=1, col=1)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.update_yaxes(
				scaleanchor = "x",
				scaleratio = 1,
			)
			fig.update_layout(
				title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}_contour.html',
				include_plotlyjs = 'cdn',
			)
		
		# Plot total collected charge...
		if False:
			total_collected_charge_average = total_collected_charge.groupby(['n_x','n_y','n_pulse']).apply(numpy.mean)
			total_collected_charge_average = total_collected_charge_average.to_frame().merge(averages[['x (m)','y (m)']].groupby(['n_x','n_y']).agg(numpy.mean), left_index=True, right_index=True)
			total_collected_charge_average = pandas.pivot_table(
				data = total_collected_charge_average,
				values = total_collected_charge_average.columns,
				index = 'y (m)',
				columns = 'x (m)',
			)
			total_collected_charge_average = total_collected_charge_average['Total collected charge (V s)']
			
			fig = px.imshow(
				total_collected_charge_average,
				title = f'Total charge as a function of position<br><sup>{bureaucrat.run_name}</sup>',
				aspect = 'equal',
				labels = dict(
					color = 'Total collected charge (V s)',
				),
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'total_collected_charge_vs_position.html',
				include_plotlyjs = 'cdn',
			)
		
		# Calculate variables...
		data = data.unstack('n_channel')
		f_horizontal = data[('Charge shared fraction',2)] + data[('Charge shared fraction',4)]
		f_vertical = data[('Charge shared fraction',1)] + data[('Charge shared fraction',2)]
		log_14 = numpy.log10(data[('Collected charge (V s)',1)]/data[('Collected charge (V s)',4)])
		log_23 = numpy.log10(data[('Collected charge (V s)',2)]/data[('Collected charge (V s)',3)])
		for _,_2 in {'f_horizontal':f_horizontal, 'f_vertical':f_vertical, 'log_14':log_14, 'log_23':log_23}.items():
			_2.name = _
		data = data.stack('n_channel')
		variables = pandas.concat([f_horizontal,f_vertical,log_14,log_23], axis=1)
		
		_ = variables.groupby(['n_pulse','n_x','n_y']).agg(numpy.nanmean)
		variables = pandas.pivot_table(
			data = _,
			values = _.columns,
			index = ['n_x','n_pulse'],
			columns = 'n_y',
		)
		
		for col in set(variables.columns.get_level_values(0)):
			df = variables.query('n_pulse==1').reset_index('n_pulse', drop=True)[col]
			print(df)
			fig = px.imshow(
				df,
				title = f'{col}<br><sup>{bureaucrat.run_name}</sup>',
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{col}.html',
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

