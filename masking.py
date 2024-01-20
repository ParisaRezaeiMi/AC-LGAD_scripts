from the_bureaucrat.bureaucrats import RunBureaucrat # https://github.com/SengerM/the_bureaucrat
from pathlib import Path
import pandas
import plotly.express as px
import logging
import numpy
import imageio.v3 as iio
import utils
import sqlite3

def create_mask_from_png(bureaucrat:RunBureaucrat, mask_file_name:str):
	bureaucrat.check_these_tasks_were_run_successfully('TCT_2D_scan')
	
	logging.info(f'Creating mask in {bureaucrat.pseudopath}')
	with bureaucrat.handle_task('create_mask_from_png') as employee:
		mask = iio.imread(bureaucrat.path_to_run_directory/mask_file_name)
		mask = numpy.mean(mask, axis=2) # Sum r,g,b channels into a single channel.
		# Convert values into 0 and 1:
		mask[mask<mask.max()] *= 0
		mask /= mask.max()
		mask = mask.astype(int)
		
		mask = numpy.flip(mask, axis=0) # PNG images are usually defined with an inverted "y" axis, so here I fix this.
		
		# Create dataframe, it is more human friendly:
		mask = pandas.DataFrame(data = mask)
		mask.index.name = 'n_y'
		mask.columns.name = 'n_x'
		
		positions_data = pandas.read_pickle(bureaucrat.path_to_directory_of_task('TCT_2D_scan')/'positions.pickle')
		
		mask = mask.stack()
		mask.name = 'mask'
		mask = positions_data.join(mask)['mask']
		
		utils.save_dataframe(
			mask,
			name = 'mask',
			location = employee.path_to_directory_of_my_task,
			formats = ['csv','pickle'],
		)
		
		# Plot the mask for future reference:
		fig = px.imshow(
			mask.reset_index('n_position', drop=True).unstack('n_x'),
			title = f'Mask<br><sup>{employee.pseudopath}</sup>',
			origin = 'lower',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'mask.html',
			include_plotlyjs = 'cdn',
		)

		# Make a plot using the data from the 2D scan and the mask, just to check it works fine...
		logging.info('Producing some plots using the mask...')
		n_positions_according_to_mask = tuple(mask[mask>0].index.get_level_values('n_position'))
		data = pandas.read_sql(
			sql = f'SELECT n_position,`Amplitude (V)` FROM dataframe_table WHERE n_position IN {tuple(n_positions_according_to_mask)} AND n_pulse==1',
			con = sqlite3.connect(bureaucrat.list_subruns_of_task('TCT_2D_scan')[0].path_to_directory_of_task('TCT_1D_scan')/'parsed_from_waveforms.sqlite'),
		).set_index('n_position')
		
		stats = data.groupby('n_position').agg(numpy.nanmean)
		stats = stats.join(positions_data.reset_index(['n_x','n_y'], drop=False))
		
		fig = px.imshow(
			stats.set_index(['n_x','n_y']).unstack('n_x')['Amplitude (V)'],
			title = f'Mask<br><sup>{employee.pseudopath}</sup>',
			origin = 'lower',
		)
		fig.write_html(
			employee.path_to_directory_of_my_task/'average_amplitude_using_mask.html',
			include_plotlyjs = 'cdn',
		)

if __name__ == '__main__':
	import argparse
	from plotly_utils import set_my_template_as_default
	import sys
	
	logging.basicConfig(
		stream = sys.stderr, 
		level = logging.INFO,
		format = '%(levelname)s|%(funcName)s|%(message)s',
		datefmt = '%Y-%m-%d %H:%M:%S',
	)
	
	set_my_template_as_default()
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',
		metavar = 'path', 
		help = 'Path to the base measurement directory.',
		required = True,
		dest = 'directory',
		type = Path,
	)
	
	args = parser.parse_args()
	create_mask_from_png(
		bureaucrat = RunBureaucrat(args.directory),
		mask_file_name = 'mask.png',
	)
