from bureaucrat.Bureaucrat import Bureaucrat # https://github.com/SengerM/bureaucrat
from pathlib import Path
import pandas
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.misc import derivative
from clean_beta_scan import binned_fit_langauss
from landaupy import langauss, landau # https://github.com/SengerM/landaupy
from grafica.plotly_utils.utils import scatter_histogram # https://github.com/SengerM/grafica

def hex_to_rgba(h, alpha):
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def script_core(directory: Path):
	John = Bureaucrat(
		directory,
		variables = locals(),
	)
	
	try:
		measured_data_df = pandas.read_feather(John.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.fd'))
	except FileNotFoundError:
		measured_data_df = pandas.read_csv(John.processed_by_script_dir_path('beta_scan.py')/Path('measured_data.csv'))
	
	for col in {'n_trigger'}:
		measured_data_df[col] = measured_data_df[col].astype(int)
	
	interesting_columns = set()
	for column in measured_data_df:
		if column in {'n_trigger','When','device_name'}: 
			continue
		interesting_columns.add(column)
		fig = px.histogram(
			measured_data_df,
			x = column,
			color = 'device_name',
			barmode = 'overlay',
			opacity = .75,
			title = f'{column}<br><sup>Measurement: {John.measurement_name}</sup>',
			marginal = 'rug',
		)
		fig.write_html(
			str(John.processed_data_dir_path/Path(f'{column} histogram.html')),
			include_plotlyjs = 'cdn',
		)
		fig = px.ecdf(
			measured_data_df,
			x = column,
			color = 'device_name',
			title = f'{column}<br><sup>Measurement: {John.measurement_name}</sup>',
			marginal = 'rug',
		)
		fig.write_html(
			str(John.processed_data_dir_path/Path(f'{column} ECDF.html')),
			include_plotlyjs = 'cdn',
		)
	
	columns_for_scatter_matrix_plot = interesting_columns - {f't_{i*10} (s)' for i in [1,2,3,4,6,7,8,9]}
	columns_for_scatter_matrix_plot = columns_for_scatter_matrix_plot - {'Humidity (%RH)','Temperature (°C)','Bias voltage (V)','Bias current (A)'}
	df = measured_data_df
	fig = px.scatter_matrix(
		df,
		dimensions = sorted(columns_for_scatter_matrix_plot),
		title = f'Scatter matrix plot<br><sup>Measurement: {John.measurement_name}</sup>',
		color = 'device_name',
		hover_data = ['n_trigger'],
	)
	fig.update_traces(diagonal_visible=False, showupperhalf=False)
	for k in range(len(fig.data)):
		fig.data[k].update(
			selected = dict(
				marker = dict(
					opacity = 1,
					color = 'black',
				)
			),
			# ~ unselected = dict(
				# ~ marker = dict(
					# ~ opacity = 0.01
				# ~ )
			# ~ ),
		)
	fig.write_html(
		str(John.processed_data_dir_path/Path('scatter matrix')) + '.html',
		include_plotlyjs = 'cdn',
	)
	
	# Fit a Landau to the collected charge ---
	for column in measured_data_df.columns:
		if 'collected charge' not in column.lower():
			continue
		fig = go.Figure()
		fig.update_layout(
			title = f'Langauss fit<br><sup>Measurement: {John.measurement_name}</sup>',
			xaxis_title = column,
			yaxis_title = 'Probability density',
		)
		colors = iter(px.colors.qualitative.Plotly)
		for device_name in sorted(set(measured_data_df['device_name'])):
			samples_for_langauss_fit = measured_data_df.query(f'device_name=={repr(device_name)}')[column].dropna()
			popt, _, hist, bin_centers = binned_fit_langauss(samples_for_langauss_fit)
			this_channel_color = next(colors)
			fig.add_trace(
				scatter_histogram(
					samples = samples_for_langauss_fit,
					bins = list(bin_centers - np.diff(bin_centers)[0]) + [bin_centers[-1]+np.diff(bin_centers)[-1]],
					name = f'Data {device_name}',
					error_y = dict(type='auto'),
					legendgroup = device_name,
					density = True,
					line = dict(color = this_channel_color),
				)
			)
			x_axis = np.linspace(min(bin_centers),max(bin_centers),999)
			fig.add_trace(
				go.Scatter(
					x = x_axis,
					y = langauss.pdf(x_axis, *popt),
					name = f'Langauss fit {device_name}<br>x<sub>MPV</sub>={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
					line = dict(color = this_channel_color, dash='dash'),
					legendgroup = device_name,
				)
			)
			fig.add_trace(
				go.Scatter(
					x = x_axis,
					y = landau.pdf(x_axis, popt[0], popt[1]),
					name = f'Landau component {device_name}',
					line = dict(color = f'rgba{hex_to_rgba(this_channel_color, .4)}', dash='dashdot'),
					legendgroup = device_name,
				)
			)
		fig.write_html(
			str(John.processed_data_dir_path/Path(f'{column} langauss fit.html')),
			include_plotlyjs = 'cdn',
		)
	
	# ~ # Plot waveforms ---
	# ~ try:
		# ~ waveforms_df = pandas.read_feather(John.processed_by_script_dir_path('beta_scan.py')/Path('waveforms.fd'))
	# ~ except FileNotFoundError:
		# ~ waveforms_df = pandas.read_csv(John.processed_by_script_dir_path('beta_scan.py')/Path('waveforms.csv'))
	
	# ~ fig = px.line(
		# ~ waveforms_df,
		# ~ x = 'Time (s)',
		# ~ y = 'Amplitude (V)',
		# ~ facet_row = 'device_name',
		# ~ line_group = 'n_trigger',
		# ~ render_mode = 'webgl', # https://plotly.com/python/webgl-vs-svg/
	# ~ )
	# ~ fig.write_html(
		# ~ str(John.processed_data_dir_path/Path('waveforms.html')),
		# ~ include_plotlyjs = 'cdn',
	# ~ )
	
########################################################################

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='Makes plots with the distributions of the quantities parsed by the script "parse_raw_data_of_single_beta_scan.py".')
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = str,
	)

	args = parser.parse_args()
	script_core(Path(args.directory))
