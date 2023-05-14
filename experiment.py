from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
from huge_dataframe.SQLiteDataFrame import load_whole_dataframe # https://github.com/SengerM/huge_dataframe
import plotly.express as px
import plotly.graph_objects as go
from grafica.plotly_utils.utils import set_my_template_as_default
import numpy
import scipy.ndimage as ndimage

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

def filter_nan_gaussian_conserving(arr, sigma):
	# https://stackoverflow.com/a/61481246/8849755
	"""Apply a gaussian filter to an array with nans.

	Intensity is only shifted between not-nan pixels and is hence conserved.
	The intensity redistribution with respect to each single point
	is done by the weights of available pixels according
	to a gaussian distribution.
	All nans in arr, stay nans in gauss.
	"""
	nan_msk = numpy.isnan(arr)

	loss = numpy.zeros(arr.shape)
	loss[nan_msk] = 1
	loss = ndimage.gaussian_filter(
			loss, sigma=sigma, mode='constant', cval=1)

	gauss = arr.copy()
	gauss[nan_msk] = 0
	gauss = ndimage.gaussian_filter(
			gauss, sigma=sigma, mode='constant', cval=0)
	gauss[nan_msk] = numpy.nan

	gauss += loss * arr

	return gauss

def filter_nan_gaussian_conserving2(arr, sigma):
	# https://stackoverflow.com/a/61481246/8849755
	"""Apply a gaussian filter to an array with nans.

	Intensity is only shifted between not-nan pixels and is hence conserved.
	The intensity redistribution with respect to each single point
	is done by the weights of available pixels according
	to a gaussian distribution.
	All nans in arr, stay nans in gauss.
	"""
	nan_msk = numpy.isnan(arr)

	loss = numpy.zeros(arr.shape)
	loss[nan_msk] = 1
	loss = ndimage.gaussian_filter(
			loss, sigma=sigma, mode='constant', cval=1)

	gauss = arr / (1-loss)
	gauss[nan_msk] = 0
	gauss = ndimage.gaussian_filter(
			gauss, sigma=sigma, mode='constant', cval=0)
	gauss[nan_msk] = numpy.nan

	return gauss

def filter_nan_gaussian_david(arr, sigma):
	# https://stackoverflow.com/a/61481246/8849755
    """Allows intensity to leak into the nan area.
    According to Davids answer:
        https://stackoverflow.com/a/36307291/7128154
    """
    gauss = arr.copy()
    gauss[numpy.isnan(gauss)] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode='constant', cval=0)

    norm = numpy.ones(shape=arr.shape)
    norm[numpy.isnan(arr)] = 0
    norm = ndimage.gaussian_filter(
            norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = numpy.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[numpy.isnan(arr)] = numpy.nan
    return gauss

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
		
		data = parsed_from_waveforms
		data.reset_index('n_waveform', drop=True, inplace=True)
		
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
		for _ in {'Amplitude','Charge'}:
			variables[f'f_{_.lower()}_horizontal'] = data[(f'{_} shared fraction',1)] + data[(f'{_} shared fraction',3)] - data[(f'{_} shared fraction',2)] - data[(f'{_} shared fraction',4)]
			variables[f'f_{_.lower()}_vertical'] = data[(f'{_} shared fraction',1)] + data[(f'{_} shared fraction',2)] - data[(f'{_} shared fraction',3)] - data[(f'{_} shared fraction',4)]
		variables['log_14'] = numpy.log10(data[('Collected charge (V s)',1)]/data[('Collected charge (V s)',4)])
		variables['log_23'] = numpy.log10(data[('Collected charge (V s)',2)]/data[('Collected charge (V s)',3)])
		for n_channel in [2,3,4]:
			variables[f'Time from CH1 of CH{n_channel} (s)'] = data[('Time from CH1 (s)',n_channel)]
		for n_channel in [1,2,3,4]:
			variables[f'Amplitude CH{n_channel} (V)'] = data[('Amplitude (V)',n_channel)]
			variables[f'Collected charge CH{n_channel} (V s)'] = data[('Collected charge (V s)',n_channel)]
		for _,_2 in variables.items():
			_2.name = _
		variables = pandas.concat([item for _,item in variables.items()], axis=1)
		data = data.stack('n_channel') # Revert what I have done before.
		
		# Add xy position information:
		variables = variables.merge(positions_data[['n_x','n_y']], left_index=True, right_index=True)
		variables.set_index(['n_x','n_y'], append=True, inplace=True)
		
		# Get rid of `n_pulse` since I will not use it here:
		variables = variables.query('n_pulse==1')
		variables.reset_index('n_pulse',drop=True,inplace=True)
		
		# Calculate statistics position-wise and create 2D table:
		variables = variables.groupby(['n_x','n_y']).agg([('average',numpy.nanmedian),('fluctuations',numpy.nanstd)])
		variables = pandas.pivot_table(
			data = variables,
			values = variables.columns,
			index = 'n_x',
			columns = 'n_y',
		)
		
		# Calculate `thing` for each variable:
		_things = []
		for variable_name in variables.columns.get_level_values(0).drop_duplicates():
			thing = calculate_thing(variables[(variable_name,'average')],variables[(variable_name,'fluctuations')],positions_data)
			_things.append(
				dict(
					variable = variable_name,
					thing = thing,
				)
			)
		thing = pandas.concat(
			[_['thing'] for _ in _things],
			keys = [(_['variable']) for _ in _things],
			axis = 0,
		)
		thing.index.set_names(['variable'], level=[0], inplace=True)
		
		# Do plots:
		for variable_name in variables.columns.get_level_values(0).drop_duplicates():
			df = variables[variable_name]
			for stat in df.columns.get_level_values(0).drop_duplicates():
				df2 = df[stat]
				df2.set_index(pandas.Index(sorted(set(positions_data['x (m)']))), inplace=True)
				df2 = df2.T.set_index(pandas.Index(sorted(set(positions_data['y (m)']))), 'y (m)').T
				df2.index.name = 'x (m)'
				df2.columns.name = 'y (m)'
				fig = px.imshow(
					df2.T,
					title = f'{variable_name}<br><sup>{bureaucrat.run_name}</sup>',
					aspect = 'equal',
					labels = dict(
						color = f'{variable_name} {stat}',
					),
					origin = 'lower',
				)
				fig.update_coloraxes(colorbar_title_side='right')
				fig.write_html(
					employee.path_to_directory_of_my_task/f'{variable_name} {stat}.html',
					include_plotlyjs = 'cdn',
				)
		
		for variable_name,df in thing.groupby('variable'):
			df.set_index(pandas.Index(sorted(set(positions_data['x (m)']))), inplace=True)
			df = df.T.set_index(pandas.Index(sorted(set(positions_data['y (m)']))), 'y (m)').T
			df.index.name = 'x (m)'
			df.columns.name = 'y (m)'
			df = df.T
			fig = px.imshow(
				df,
				title = f'Thing({variable_name})<br><sup>{bureaucrat.run_name}</sup>',
				aspect = 'equal',
				labels = dict(
					color = f'Thing({variable_name}) (m)',
				),
				origin = 'lower',
				zmin = 0,
				zmax = 33e-6,
			)
			fig.update_coloraxes(colorbar_title_side='right')
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{variable_name} thing.html',
				include_plotlyjs = 'cdn',
			)
			
			fig = go.Figure(
				data = go.Contour(
					z = filter_nan_gaussian_david(df, sigma=2),
					x = df.columns,
					y = df.index,
					contours = dict(
						# ~ coloring ='heatmap',
						showlabels = True,
						labelfont = dict( # label font properties
								size = 12,
								color = 'white',
						),
						start = 0,
						end = 50e-6,
						size = 2.5e-6,
					),
					line_smoothing = 1,
					colorbar = dict(
						title = f'Thing({variable_name}) (m)',
						titleside = 'right',
					),
				),
			)
			fig.update_layout(
				title = f'Thing({variable_name})<br><sup>{bureaucrat.run_name}</sup>',
				xaxis_title = df.columns.name,
				yaxis = dict(
					scaleanchor = 'x',
					title = df.index.name,
				),
			)
			fig.write_html(
				employee.path_to_directory_of_my_task/f'{variable_name} thing contour.html',
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

