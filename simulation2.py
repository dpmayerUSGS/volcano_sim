#Built-in Libraries
import math
from random import uniform
from random import randrange
import argparse
import os
import string
import ctypes

#external libraries
import numpy
import ogr
import osr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import Image
from matplotlib.image import imread
from mpl_toolkits.basemap import Basemap
from osgeo import gdal

#Constants / Globals
global velocity
global angle
global num

def create_shapefile(xdata, ydata, shapefile):
	output = shapefile[0]
	driverName = "ESRI Shapefile"
	drv = ogr.GetDriverByName(driverName)
	ds = drv.CreateDataSource(output)

	layer = ds.CreateLayer("point_out",geom_type=ogr.wkbPoint)

	#Write fields	
	field_x = ogr.FieldDefn()
	field_x.SetName('xCoord')
	field_x.SetType(ogr.OFTReal)
	field_x.SetWidth(15)
	field_x.SetPrecision(6)
	layer.CreateField(field_x)
	
	field_y = ogr.FieldDefn()
	field_y.SetName('yCoord')
	field_y.SetType(ogr.OFTReal)
	field_y.SetWidth(15)
	field_y.SetPrecision(6)
	layer.CreateField(field_y)
	
	field_itnum = ogr.FieldDefn()
	field_itnum.SetName('IterNum')
	field_itnum.SetType(ogr.OFTInteger)
	field_itnum.SetWidth(10)
	field_itnum.SetPrecision(1)
	layer.CreateField(field_itnum)
	
	#Iterate over the coordinate arrays and write the row
	for index in range(len(xdata+1)):
		feat = ogr.Feature(layer.GetLayerDefn())
		feat.SetField('IterNum', index)
		feat.SetField('xCoord', xdata[index])
		feat.SetField('yCoord', ydata[index])
		pt = ogr.Geometry(ogr.wkbPoint)
		pt.AddPoint_2D(xdata[index], ydata[index])
		feat.SetGeometry(pt)
		layer.CreateFeature(feat)
		
	spatialRef = osr.SpatialReference()
	spatialRef.SetGeogCS("GCS_Moon_2000", 
                     "D_Moon_2000", 
                     "Moon_localradius",1737400.0, 0.0,  
                     "Prime Meridian",0.0,
                     "Degree",0.0174532925199433 )
	
	
	#Output the .prj file.
	spatialRef.MorphToESRI()
	basename = output.split('.')[0]
	file = open(basename + ".prj", 'w')
	file.write(spatialRef.ExportToWkt())
	file.close()

def init(xarr_, yarr_):
	global xarr
	global yarr
	xarr = xarr_
	yarr = yarr_

def f(v):
	f = v
	return f #Using return at the end of a def statement passes the variable back to the calling function.

def random_azimuth():
	'''This function returns a random floating point number between 1 and 360'''
	#use normalvariate(mean, std) for a gaussian distribution
	#A more complex weighting can be achieved, but would need to be modeled.
	return uniform(0,360)

def strom_multi(xarr,yarr,i):
	
	for index in range(len(xarr[i])):
		
		#distance and coordinates
		distance, angle, elevation = calc_distance()
		azimuth = random_azimuth()
		Xcoordinate = distance * math.sin(azimuth * math.pi/180) #Conversion to radians
		Ycoordinate = distance * math.cos(azimuth* math.pi/180)

		#The WAC visible spectrum data is 100mpp or 0.003297790480378 degrees / pixel.
		Xcoordinate /= 100
		Xcoordinate *= 0.003297790480378
		Ycoordinate /= 100
		Ycoordinate *= 0.003297790480378
		x = float(Xcoordinate)
		y = float(Ycoordinate)
		#Randomly select the origin point along the linear vent
		rand_index = randrange(0,10)
		xorigin, yorigin = (xpt[rand_index], ypt[rand_index])
		distance = check_topography(dtm, xorigin, yorigin, x+xorigin, y+yorigin, distance,elevation, dev, gtinv)

		if distance[1] == True:
			x = (distance[0] * math.sin(azimuth * math.pi/180))
			y = (distance[0] * math.cos(azimuth* math.pi/180))
			#Convert back to degrees
			x /= 100
			x *= 0.003297790480378
			y /= 100
			y *= 0.003297790480378
			
		else:
			pass
		xarr[i][index] = x+xorigin
		yarr[i][index] = y+yorigin

def calc_height(distance, ejectionangle, g, ejectionvelocity):
	'''
	height@x = initital_height + distance(tan(theta)) - ((g(x^2))/(2(v(cos(theta))^2))

	initial_height = 0, a planar surface is fit to some reference elevation.
	
	distance is in meters
	angle is in radians
	'''
	trajectory = numpy.linspace(0,distance, distance/100,endpoint=True )
	elevation = (trajectory * math.tan(ejectionangle)) - ((g*(trajectory**2)) / (2*((ejectionvelocity * math.cos(ejectionangle))**2))) 
	return elevation
	
def calc_distance():
	g = 1.6249
	#Calculate the ejection angle randomly from a range
	ejectionangle = uniform(angle[0],angle[1])
	ejectionangle *= math.pi/180 #Convert to radians
	theta = math.sin(2*ejectionangle)
	#Determine the ejection velocity randomly from a range
	ejectionvelocity = uniform(velocity[0], velocity[1])
	v2 = ejectionvelocity * ejectionvelocity
	#Calculate total theoretical travel distance
	distance = (v2 * theta) / g
	#Calculate the elevation over a planar surface
	elevation = calc_height(distance, ejectionangle, g, ejectionvelocity)
	return distance, ejectionangle, elevation
	
def stromboli2():
	'''distance = (velocity^2*(sin(2theta))) / gravity'''
	p = 0
	while p <= num:
		p+=1
		g = 1.6249 #Gravitational acceleration on the moon
		
		distance, angle, elevation = calc_distance()
		azimuth = random_azimuth()
		Xcoordinate = distance * math.sin(azimuth * math.pi/180) #Conversion to radians
		Ycoordinate = distance * math.cos(azimuth* math.pi/180)
		#The WAC visible spectrum data is 100mpp or 0.003297790480378 degrees / pixel.
		Xcoordinate /= 100
		Xcoordinate *= 0.003297790480378
		Ycoordinate /= 100
		Ycoordinate *= 0.003297790480378
		
		yield Xcoordinate, Ycoordinate, angle, azimuth, elevation, distance
		
		if p > num:
			done = False
			yield done		

def check_topography(dtm, originx, originy, destx, desty, distance,elevation, dev, gtinv):
	'''
	This function checks for impact due to variation in topography by 
	mimicing the functionality of a topographic profile from polyline.

	1. Generate 2 arrays.  One of X coordinates and one of Y coordinates
	2. Transform these from GCS to PCS
	3. Create a new array with the elevations extracted from the dtm
	4. Compare it to the analytical trajectory heights
	5. If the impact occurs before total potential travel distance, 
	drop the projectile there.  If not, place it at the total possible 
	travel distance.
	
	Parameters
        ----------
	dtm: A digital terrain model, in 16bit, storing terrain elevation, ndarray
	originx: The x coord of the projectile launch, scalar
	originy: The y coord of the projectile launch, scalar
	destx: The x landing coordinate on a flat plane, scalar
	desty: The y landing coordinate on a flat plane, scalar
	distance: The total possible distance traveled, scalar
	elevation: An array storing heights above 0 of the projectile at some 
	interval (100m by default)
	dev: Geotransform parameters
	gtinv: Inverse geotransform parameters

	Returns
	-------
	distance: The new distance the projectile has traveled if it impacts 
	the topography.

	
	'''
	#Convert from GCS to pixel space using a nearest neighbor interpolation
	xpt = numpy.linspace(originx,destx,num=(distance)/100, endpoint=True)
	ypt = numpy.linspace(originy,desty,num=(distance)/100, endpoint=True)
	xpt -= geotransform[0]
	ypt -= geotransform[3]
	xsam = numpy.round_((gtinv[1] *xpt + gtinv[2] * ypt), decimals=0)
	ylin = numpy.round_((gtinv[4] *xpt + gtinv[5] * ypt), decimals=0)


	try:
		#Extract the elevation from the dtm along the vector
		dtmvector = dtm[ylin.astype(int),xsam.astype(int)]
		#Compute elevation of projectile from a plane at the origin height
		dtmvectormin = dtmvector.min()
		elevation -= abs(dtmvector[0])
		#Compare the projectile elevation to the dtm
		dtmvector += abs(dtmvectormin)
		elevation -= dtmvector
		elevation += dtmvectormin
		#Ignore the first 2km of ejection distance to ensure that we get a valid elevation check.
		impact =  numpy.where(elevation[200:] <= 0)
		try:
			#We are working at 100mpp, so the new distance is index +1
			#print elevation
			return ((impact[0][0])+1) * 100, True
		except:
			print "The particle does not have sufficient angle to escape."
			#print elevation
	except:
		print "Total distance travel exceeds model dimensions."
		return None, True


def density(m, xdata, ydata, shapefile, ppg):
	'''
	This function converts the lat/lon of the input map to meters 
	assuming an equirectangular projection.  It then creates a grid at 
	100mpp, bins the input data into the grid  (density) and creates a 
	histogram.  Finally, a mesh grid is created and the histogram is 
	plotted in 2D over the basemap.
	
	If the shapefile flag is set to true a shapefile is created by calling 
	the shapefile function.

	Parameters:
	m: A basemap mapping object
	xdata: An array of x landing coordinates, ndarray
	ydata: An array of y landing coordinates, ndarray
	shapefile: A flag on whether or not to generate a shapefile
	ppg: The number of meters per grid cell * 100
	
	'''
	#Convert from DD to m to create a mesh grid.
	xmax = (m.xmax) / 0.003297790480378
	xmin = (m.xmin) / 0.003297790480378
	ymax = (m.ymax) / 0.003297790480378
	ymin = (m.ymin) / 0.003297790480378
	
	#Base 100mpp 
	nx = 1516 / int(ppg)
	ny = 2123 / int(ppg)
	
	#Convert to numpy arrays
	xdata = numpy.asarray(xdata)
	ydata = numpy.asarray(ydata)
	
	#Bin the data & calculate the density
	lon_bins = numpy.linspace(xdata.min(), xdata.max(), nx+1)
	lat_bins = numpy.linspace(ydata.min(), ydata.max(), ny+1)
	density, _, _ = numpy.histogram2d(ydata, xdata, [lat_bins, lon_bins])

	#If the user wants a shapefile, pass the numpy arrays
	if shapefile != None:
		print "Writing model output to a shapefile."
		create_shapefile(xdata, ydata, shapefile)
		
	#Create a grid of equally spaced polygons
	lon_bins_2d, lat_bins_2d = numpy.meshgrid(lon_bins, lat_bins)
	if density.max() <= 3:
		maxden = 5
	else: 
		maxden = density.max()

	#Mask the density array so that 0 is not plotted
	density = numpy.ma.masked_where(density <=0, density)
	
	plt.pcolormesh(lon_bins_2d,lat_bins_2d, density, cmap=cm.RdYlGn_r, vmin=0, vmax=maxden, alpha=0.5)
	plt.colorbar(orientation='horizontal')

if __name__ == '__main__': 
	'''This is the main section which handles program flow.'''
	
	#Parse all of the arguments.
	parser = argparse.ArgumentParser(description='Stromboli Ejection Simulation Tool v1')
	parser.add_argument('--velocity', '-v', action='store',nargs='+',default=[350,425], dest='velocity', help='A range of ejection velocities. ')
	parser.add_argument('--angle','-a', action='store', nargs='+',default=[30, 60], dest='angle', help='Optional: A range of ejection angles.  Example: -a 30 60')
	parser.add_argument('-i', '--iterations', action='store', type=int, dest='i',default=500, help='The number of ejection iterations to perform.')
	parser.add_argument('--shapefile', action='store',nargs=1, default=None, dest='shapefile', help='Use this flag to generate a shapefile, in Moon_2000GCS, of the point data.')
	parser.add_argument('--fast', action='store', default=None, nargs=1, dest='multi', help='Use this flag to forgo creating a visualization and just create a shapefile.  This uses all available processing cores and is substantially faster.')
	parser.add_argument('--ppg', action='store', default=10, dest='ppg', help='The number of pixels per grid cell.  Default is 10, which generates a 1000m grid square using 100mpp WAC Vis.')
	args = parser.parse_args()
	
	#Assign the user variables to the globals, not great form, but it works.
	try:
		velocity = [float(args.velocity[0]),float(args.velocity[1])]
	except:
		velocity = [float(args.velocity[0]),float(args.velocity[0])]
	num = args.i
	try:
		angle = [float(args.angle[0]),float(args.angle[1])]
	except:
		angle = [float(args.angle[0]),float(args.angle[0])]
		
	#Read the input DTM and get geotransformation info
	ds = gdal.Open('wac_dtm.tif')
	dtm = ds.ReadAsArray()
	geotransform = ds.GetGeoTransform()
	dev = (geotransform[1]*geotransform[5] - geotransform[2]*geotransform[4])
	gtinv = ( geotransform[0] , geotransform[5]/dev, - geotransform[2]/dev, geotransform[3], - geotransform[4]/dev, geotransform[1]/dev)
	
	#Set the approximate ejection coordinates
	xpt = numpy.linspace(-97.788,-97.855,num=10, endpoint=True)
	ypt = numpy.linspace(-30.263,-29.851,num=10, endpoint=True)
	
	#If the user wants to process quickly then we omit the visualization and multiprocess to generate a shapefile
	if args.multi is not None:
		import multiprocessing
		cores = multiprocessing.cpu_count()
		cores *= 2
		step = num // cores
		xarray = numpy.frombuffer(multiprocessing.RawArray(ctypes.c_double, num))
		yarray = numpy.frombuffer(multiprocessing.RawArray(ctypes.c_double, num))
		init(xarray,yarray)
		jobs = []
		for i in range(0, num+1, step):
			p = multiprocessing.Process(target=strom_multi, args=(xarr,yarr,slice(i, i+step)), )
			jobs.append(p)
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()
			
		create_shapefile(xarr, yarr, args.multi)
	

	else:
		#Visualization - setup the plot
		fig = plt.figure(figsize=(15,10))
		ax1 = fig.add_subplot(1,2,1)
		#Points that hit underlying topography
		pt, = ax1.plot([], [],'co', markersize=3)
		xdata, ydata = [], []
		#Points that travel the total theoretical distance
		ptmax,  = ax1.plot([],[], 'yo', markersize=3)
		datamax, ydatamax = [],[]
	
		#Map
		lon_min = -102.5
		lon_max = -93.5
		lat_min = -34.5
		lat_max = -25.5
	
		m = Basemap(projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,
		    llcrnrlon=lon_min,urcrnrlon=lon_max,resolution=None, rsphere=(1737400.0,1737400.0))
		m.drawmeridians(numpy.arange(lon_min+0.5, lon_max+1, 1), labels=[0,0,0,1], fontsize=10)
		m.drawparallels(numpy.arange(lat_min+0.5,lat_max+1, 1), labels=[1,0,0,0], fontsize=10)
		
		#Read the input image
		im = imread('wac_global_vis.png')
		m.imshow(im, origin='upper', cmap=cm.Greys_r, alpha=0.9)
				
		def run(data):
			if data == False:
				density(m2,xdata, ydata, args.shapefile, args.ppg)
			else:
				#x, y are in degrees from the false origin 0,0
				x,y, angle, azimuth, elevation, distance = data
				rand_index = randrange(0,10)
				#Randomly select the origin point along the linear vent
				xorigin, yorigin = (xpt[rand_index], ypt[rand_index])
				distance = check_topography(dtm, xorigin, yorigin, x+xorigin, y+yorigin, distance,elevation, dev, gtinv)
				try:
					if distance[1] == True:
						x = (distance[0] * math.sin(azimuth * math.pi/180))
						y = (distance[0] * math.cos(azimuth* math.pi/180))
	
						#Convert back to degrees
						x /= 100
						x *= 0.003297790480378
						y /= 100
						y *= 0.003297790480378
						xdata.append(x + xorigin)
						ydata.append(y + yorigin)
						pt.set_data(xdata, ydata)
					else:
						print 'Particle landed at the maximum theoretical distance.'
						#Convert back to degrees
						x /= 100
						x *= 0.003297790480378
						y /= 100
						y *= 0.003297790480378
						xdatamax.append(x + xorigin)
						ydatamax.append(y + yorigin)
						#Set the point
						ptmax.set_data(xdatamax, ydatamax)
					print 'Angle: %f, Azimuth: %f, xCoordinate: %f, yCoordinate: %f' %(angle, azimuth,x+xorigin,y+yorigin)
					return pt,
				except:
					pass
		
		#Plot the volcano as approximated by a linear function.
		plt.plot(xpt, ypt, 'bo', markersize=4)
		#Run the animation
		ani = animation.FuncAnimation(fig, run,stromboli2, interval=1, repeat=False, blit=False)
	
		plt.title('Sample Deposition Using ' + (str(num+2)) + " Points")
		ax2 = fig.add_subplot(1,2,2)
		gridsize = str(int(args.ppg) * 100)
		ax2.set_title('Impacts /' + gridsize + ' m') 
		
		m2 = Basemap(projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,
		    llcrnrlon=lon_min,urcrnrlon=lon_max,resolution=None, rsphere=(1737400.0,1737400.0))
		m.drawmeridians(numpy.arange(lon_min+0.5, lon_max+1, 1), labels=[0,0,0,1], fontsize=10)
		m.drawparallels(numpy.arange(lat_min+0.5,lat_max+1, 1), labels=[1,0,0,0], fontsize=10)	
		
		m2.imshow(im, origin='upper', cmap=cm.Greys_r)
		
		plt.show()
		
		#Save the animation 
		#ani.save('simulation.mp4', fps=10)
	
	


	

	