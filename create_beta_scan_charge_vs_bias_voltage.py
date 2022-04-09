import pandas
from bureaucrat.Bureaucrat import Bureaucrat
from pathlib import Path
import plotly.express as px
import numpy as np
import warnings

def get__BetaScan_sweeping_bias_voltage__list_of_fixed_voltage_scans(measurement_path: Path):
	scans = []
	with open(measurement_path/Path('beta_scan_sweeping_bias_voltage/README.txt'), 'r') as ifile:
		for idx, line in enumerate(ifile):
			if idx == 0:
				continue
			scans.append(line.replace('\n',''))
	return scans

def script_core(measurement_path: Path, force=True):
	bureaucrat = Bureaucrat(
		measurement_path,
		new_measurement = False,
		variables = locals(),
	)
	
	if force == False and bureaucrat.job_successfully_completed_by_script('this script'):
		return
	
	collected_charge_df = pandas.DataFrame()
	with bureaucrat.verify_no_errors_context():
		for measurement_name in get__BetaScan_sweeping_bias_voltage__list_of_fixed_voltage_scans(bureaucrat.measurement_base_path):
			try:
				df = pandas.read_csv(bureaucrat.measurement_base_path.parent/Path(measurement_name)/Path('calculate_collected_charge_beta_scan/results.csv'))
			except FileNotFoundError:
				warnings.warn(f'Cannot read data from measurement {repr(measurement_name)}')
				continue
			collected_charge_df = collected_charge_df.append(
				{
					'Collected charge (V s) x_mpv': float(df.query(f'`Device name`=="RSD1 #18S GuardRing Chubut"').query('Variable=="Collected charge (V s) x_mpv"').query('Type=="fit to data"')['Value']),
					'Measurement name': measurement_name,
					'Bias voltage (V)': int(measurement_name.split('_')[-1].replace('V','')),
					'Measurement name': measurement_name,
				},
				ignore_index = True,
			)
		
		df = collected_charge_df.sort_values(by='Bias voltage (V)')
		fig = px.line(
			title = f'Collected charge vs bias voltage with beta source<br><sup>Measurement: {bureaucrat.measurement_name}</sup>',
			data_frame = df,
			x = 'Bias voltage (V)',
			y = 'Collected charge (V s) x_mpv',
			hover_data = sorted(df),
			markers = 'circle',
		)
		fig.write_html(
			str(bureaucrat.processed_data_dir_path/Path('collected charge vs bias voltage.html')),
			include_plotlyjs = 'cdn',
		)
		collected_charge_df.to_csv(bureaucrat.processed_data_dir_path/Path('collected_charge_vs_bias_voltage.csv'))

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--dir',
		metavar = 'path', 
		help = 'Path to the base directory of a measurement. If "all", the script is applied to all linear scans.',
		required = True,
		dest = 'directory',
		type = str,
	)
	args = parser.parse_args()
	script_core(Path(args.directory), force=True)
