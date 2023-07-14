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
	fig = px.imshow(
		z,
		labels = dict(
			color = z_name,
		),
		**plotly_kwargs,
	)
	fig.update_coloraxes(colorbar_title_side='right')
	return fig

def plot_as_xy_contour(z:pandas.Series, positions_data:pandas.DataFrame, zmin=None, zmax=None, title=None):
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
			z = filter_nan_gaussian_david(z, sigma=2),
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
