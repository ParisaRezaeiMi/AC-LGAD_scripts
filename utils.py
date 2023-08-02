from pathlib import Path
import pandas
import plotly.express as px
import plotly.graph_objects as go
import numpy
import scipy.ndimage as ndimage

def filter_nan_gaussian_david(arr, sigma):
	# https://stackoverflow.com/a/61481246/8849755
    """Allows intensity to leak into the nan area.
    According to Davids answer:
        https://stackoverflow.com/a/36307291/7128154
    """
    gauss = arr.copy()
    gauss[numpy.isnan(gauss)] = 0
    gauss = ndimage.gaussian_filter(gauss, sigma=sigma, mode='constant', cval=0)

    norm = numpy.ones(shape=arr.shape)
    norm[numpy.isnan(arr)] = 0
    norm = ndimage.gaussian_filter(norm, sigma=sigma, mode='constant', cval=0)

    # avoid RuntimeWarning: invalid value encountered in true_divide
    norm = numpy.where(norm==0, 1, norm)
    gauss = gauss/norm
    gauss[numpy.isnan(arr)] = numpy.nan
    return gauss

def save_dataframe(df, name:str, location:Path):
	for extension,method in {'pickle':df.to_pickle,'csv':df.to_csv}.items():
		method(location/f'{name}.{extension}')

def plot_as_xy_heatmap(z:pandas.Series, positions_data:pandas.DataFrame, **plotly_kwargs):
	"""Produce a heatmap using plotly of some given quantity as a function
	of x and y.
	
	Arguments
	---------
	z: pandas.Series
		The quantity for which you want to produce the heatmap. It has
		to be a Series like the following example:
		```
		n_position
		0       0.015565
		1       0.006891
		2       0.009484
		3       0.006305
		4       0.006852
				  ...   
		3131    0.004369
		3132    0.004456
		3133    0.004636
		3134    0.004340
		3135    0.004351
		Name: Amplitude (V), Length: 3136, dtype: float64
		```
	positions_data: pandas.DataFrame
		A data frame with the information about the positions in x and y
		for each `n_position`, example:
		```
							n_x  n_y  Unnamed: 0     x (m)     y (m)     z (m)
		n_position                                                    
		0             0    0           0 -0.000275 -0.000275  0.067957
		1             1    0           1 -0.000265 -0.000275  0.067957
		2             2    0           2 -0.000255 -0.000275  0.067957
		3             3    0           3 -0.000245 -0.000275  0.067957
		4             4    0           4 -0.000235 -0.000275  0.067957
		...         ...  ...         ...       ...       ...       ...
		3131         51   55        3131  0.000235  0.000275  0.067957
		3132         52   55        3132  0.000245  0.000275  0.067957
		3133         53   55        3133  0.000255  0.000275  0.067957
		3134         54   55        3134  0.000265  0.000275  0.067957
		3135         55   55        3135  0.000275  0.000275  0.067957
		[3136 rows x 6 columns]
		```
		plotly_kwargs
			Any keyword arguments to be passed to Plotly when constructing
			the figure.
	"""
	
	if not isinstance(z, pandas.Series):
		raise TypeError(f'`z` must be an instance of {pandas.Series}, but received instead an object of type {type(z)}. ')
	
	z_name = z.name
	z = z.to_frame()
	z = z.merge(positions_data[['x (m)','y (m)','n_x','n_y']], left_index=True, right_index=True)
	
	z = pandas.pivot_table(
		data = z,
		values = z_name,
		index = 'n_x',
		columns = 'n_y',
	)
	z.set_index(
		keys = pandas.Index(sorted(set(positions_data['x (m)']))[::-1]), 
		inplace = True,
	)
	z = z.T
	z.set_index(
		pandas.Index(sorted(set(positions_data['y (m)']))),
		inplace = True,
	)
	z = z.T
	z.index.name = 'x (m)'
	z.columns.name = 'y (m)'
	z = z.T
	z = pandas.DataFrame(
		data = numpy.flip(z.to_numpy(), axis=0),
		index = z.index,
		columns = z.columns,
	)
	fig = px.imshow(
		z,
		labels = dict(
			color = z_name,
		),
		**plotly_kwargs,
	)
	fig.update_coloraxes(colorbar_title_side='right')
	return fig

def plot_as_xy_contour(z:pandas.Series, positions_data:pandas.DataFrame, zmin=None, zmax=None, title=None, smoothing_sigma=0):
	if not isinstance(z, pandas.Series):
		raise TypeError(f'`z` must be an instance of {pandas.Series}, but received instead an object of type {type(z)}. ')
	
	z_name = z.name
	z = z.to_frame()
	z = z.merge(positions_data[['x (m)','y (m)','n_x','n_y']], left_index=True, right_index=True)
	
	z = pandas.pivot_table(
		data = z,
		values = z_name,
		index = 'n_x',
		columns = 'n_y',
	)
	z.set_index(
		keys = pandas.Index(sorted(set(positions_data['x (m)']))[::-1]), 
		inplace = True,
	)
	z = z.T
	z.set_index(
		pandas.Index(sorted(set(positions_data['y (m)']))),
		inplace = True,
	)
	z = z.T
	z.index.name = 'x (m)'
	z.columns.name = 'y (m)'
	z = z.T
	fig = go.Figure(
		data = go.Contour(
			z = filter_nan_gaussian_david(z.to_numpy(), sigma=smoothing_sigma),
			x = z.columns,
			y = z.index,
			contours = dict(
				# ~ coloring ='heatmap',
				showlabels = True,
				labelfont = dict( # label font properties
						size = 12,
						color = 'white',
				),
				start = zmin,
				end = zmax,
				# ~ size = 2.5e-6,
			),
			line_smoothing = 1,
			colorbar = dict(
				title = z_name,
				titleside = 'right',
			),
		),
	)
	fig.update_layout(
		xaxis_title = z.columns.name,
		yaxis = dict(
			scaleanchor = 'x',
			title = z.index.name,
		),
		title = title,
	)
	return fig

def resample_positions(positions_data:pandas.DataFrame, new_n_x:int, new_n_y:int):
	"""Produce a new table of positions with a larger sampling. Only works
	for square 2D sampling. 
	
	Arguments
	---------
	positions_data: pandas.DataFrame
		A data frame of the form
		```
		            n_x  n_y     x (m)     y (m)
		n_position                              
		0            12    0 -0.000144 -0.000278
		1            13    0 -0.000133 -0.000278
		2            14    0 -0.000122 -0.000278
		3            15    0 -0.000111 -0.000278
		4            16    0 -0.000100 -0.000278
		...         ...  ...       ...       ...
		2020         34   50  0.000100  0.000277
		2021         35   50  0.000111  0.000277
		2022         36   50  0.000122  0.000277
		2023         37   50  0.000133  0.000277
		2024         38   50  0.000144  0.000277
		```
		with the positions to be resampled.
	new_n_x, new_n_y: int
		New number of points in each coordinate.
	
	Returns
	-------
	resampled_positions_data: pandas.DataFrame
		A data frame identical to `positions_data` with the new sampling,
		for example:
		```
		            n_x  n_y     x (m)         y (m)
		n_position                                  
		0             1    0 -0.000111 -2.220000e-04
		1             2    0  0.000000 -2.220000e-04
		2             3    0  0.000111 -2.220000e-04
		4             1    1 -0.000111 -1.110000e-04
		5             2    1  0.000000 -1.110000e-04
		6             3    1  0.000111 -1.110000e-04
		3             0    1 -0.000222 -1.110000e-04
		7             4    1  0.000222 -1.110000e-04
		8             0    2 -0.000222 -5.421011e-20
		9             1    2 -0.000111 -5.421011e-20
		10            2    2  0.000000 -5.421011e-20
		11            3    2  0.000111 -5.421011e-20
		12            4    2  0.000222 -5.421011e-20
		13            0    3 -0.000222  1.110000e-04
		14            1    3 -0.000111  1.110000e-04
		15            2    3  0.000000  1.110000e-04
		16            3    3  0.000111  1.110000e-04
		17            4    3  0.000222  1.110000e-04
		18            1    4 -0.000111  2.220000e-04
		19            2    4  0.000000  2.220000e-04
		20            3    4  0.000111  2.220000e-04
		```
	n_position_mapping: pandas.Series
		A series in which the index is the old `n_position` and the values
		are the new `n_position`, for example:
		```
		n_position
		0        0
		1        0
		2        0
		3        0
		4        0
				..
		2020    20
		2021    20
		2022    20
		2023    20
		2024    20
		Name: n_position, Length: 2025, dtype: int64
		```
	"""
	dxy = {}
	xy_start = {}
	xy_stop = {}
	for xy,n in {'x':new_n_x,'y':new_n_y}.items():
		minval = numpy.nanmin(positions_data[f'{xy} (m)'])
		maxval = numpy.nanmax(positions_data[f'{xy} (m)'])
		dxy[xy] = (maxval-minval)/(n)
		xy_start[xy] = minval
		xy_stop[xy] = maxval
	
	dx_old = numpy.diff(positions_data['x (m)'].drop_duplicates())[0]
	dy_old = numpy.diff(positions_data['y (m)'].drop_duplicates())[0]
	
	resampled_positions_data = positions_data.copy()
	new_n_position = -1
	y_boundaries = numpy.arange(start=xy_start['y'],stop=xy_stop['y'],step=dxy['y'])
	x_boundaries = numpy.arange(start=xy_start['x'],stop=xy_stop['x'],step=dxy['x'])
	for new_ny,new_y in enumerate(y_boundaries):
		for new_nx,new_x in enumerate(x_boundaries):
			new_n_position += 1
			for key,val in {'n_x':new_nx, 'n_y':new_ny, 'n_position':new_n_position, 'x (m)':new_x+dxy['x']/2, 'y (m)':new_y+dxy['y']/2}.items():
				resampled_positions_data.loc[(positions_data['x (m)']>=new_x)&(positions_data['x (m)']<=new_x+dxy['x']+dx_old*1.1)&(positions_data['y (m)']>=new_y)&(positions_data['y (m)']<=new_y+dxy['y']+dy_old*1.1), key] = val
	resampled_positions_data = resampled_positions_data.astype({'n_x':int,'n_y':int,'n_position':int})
	resampled_positions_data.index.rename('old_n_position', inplace=True)
	
	n_position_fixed = 0
	for n_position in sorted(resampled_positions_data['n_position'].drop_duplicates()):
		_ = resampled_positions_data.loc[resampled_positions_data['n_position']==n_position]
		resampled_positions_data.loc[resampled_positions_data['n_position']==n_position,'n_position'] = n_position_fixed
		if len(_) > 0:
			n_position_fixed += 1
	
	n_position_mapping = resampled_positions_data['n_position']
	n_position_mapping.index.rename('n_position',inplace=True)
	resampled_positions_data.set_index('n_position', inplace=True)
	resampled_positions_data = resampled_positions_data.drop_duplicates()
	return resampled_positions_data, n_position_mapping
