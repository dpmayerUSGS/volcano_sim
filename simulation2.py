#Built-in Libraries
import math
from random import uniform
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
from matplotlib.image import imread
from mpl_toolkits.basemap import Basemap



#Constants / Globals
global velocity
global num

INTERVALS = 100000.0
G = -1.62
TNOT = 0.0 #Initial Time
TEND = 1000.0 # End Time
H = (TEND - TNOT) / INTERVALS
Z0 = 0 

def create_shapefile(xdata, ydata):
	output = 'test2.shp'
	driverName = "ESRI Shapefile"
	drv = ogr.GetDriverByName(driverName)
	ds = drv.CreateDataSource(output)

	layer = ds.CreateLayer("point_out",geom_type=ogr.wkbPoint)

	#Write fields	
	field_x = ogr.FieldDefn()
	field_x.SetName('xCoord')
	field_x.SetType(ogr.OFTReal)
	field_x.SetWidth(10)
	field_x.SetPrecision(6)
	layer.CreateField(field_x)
	
	field_y = ogr.FieldDefn()
	field_y.SetName('yCoord')
	field_y.SetType(ogr.OFTReal)
	field_y.SetWidth(10)
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
	return uniform(1,360)

def strom_multi(xarr,yarr,i):
	
	for index in range(len(xarr[i])):
		v0 = velocity
		angle = uniform(30, 60) #Setup for a random angle between 30 and 60 deg. 
		
		vz0 = v0 * math.sin(angle * math.pi / 180) #Vertical component of the verlocity
		vx0 = v0 * math.cos(angle * math.pi / 180) #Horiz component of the vel.
		t = TNOT 
		z = Z0 
		v = vz0 
		
		#Runge-Kutta Routine ''''''
		while z >= 0:
			k1 = H * f(v) #The vertical component of the vel * step size
			l1 = H * G            
			k2 = H * f(v + l1 / 2.)
			l2 = H * G
			k3 = H * f(v + l2 / 2.)
			l3 = H * G
			k4 = H * f(v + l3)
			l4 = H * G
			znew = z + (k1 + 2. * (k2 + k3) + k4) / 6.
			vnew = v + (l1 + 2. * (l2 + l3) + l4) / 6.
			t = t + H
			z = znew
			v = vnew
	
			
		#distance and coordinates
		distance = t * vx0
		azimuth = random_azimuth()
		Xcoordinate = distance * math.sin(azimuth * math.pi/180) #Conversion to radians
		Ycoordinate = distance * math.cos(azimuth* math.pi/180)

		#The WAC visible spectrum data is 100mpp or 0.003297790480378 degrees / pixel.
		Xcoordinate /= 100
		Xcoordinate *= 0.003297790480378
		Ycoordinate /= 100
		Ycoordinate *= 0.003297790480378
		xorigin, yorigin = (-97.7328, -30.0906, ) #This is an estimate
		Xcoordinate += xorigin
		Ycoordinate += yorigin
		xarr[i][index] = Xcoordinate
		yarr[i][index] = Ycoordinate
		
def stromboli():
	p = 0
	while p <= num:
		p+=1
		v0 = velocity
		angle = uniform(30, 60) #Setup for a random angle between 30 and 60 deg. 
		
		vz0 = v0 * math.sin(angle * math.pi / 180) #Vertical component of the verlocity
		vx0 = v0 * math.cos(angle * math.pi / 180) #Horiz component of the vel.
		t = TNOT 
		z = Z0 
		v = vz0 
		
		#Runge-Kutta Routine ''''''
		while z >= 0:
			k1 = H * f(v) #The vertical component of the vel * step size
			l1 = H * G            
			k2 = H * f(v + l1 / 2.)
			l2 = H * G
			k3 = H * f(v + l2 / 2.)
			l3 = H * G
			k4 = H * f(v + l3)
			l4 = H * G
			znew = z + (k1 + 2. * (k2 + k3) + k4) / 6.
			vnew = v + (l1 + 2. * (l2 + l3) + l4) / 6.
			t = t + H
			z = znew
			v = vnew
			
		#distance and coordinates
		distance = t * vx0
		azimuth = random_azimuth()
		Xcoordinate = distance * math.sin(azimuth * math.pi/180) #Conversion to radians
		Ycoordinate = distance * math.cos(azimuth* math.pi/180)

		#The WAC visible spectrum data is 100mpp or 0.003297790480378 degrees / pixel.
		Xcoordinate /= 100
		Xcoordinate *= 0.003297790480378
		Ycoordinate /= 100
		Ycoordinate *= 0.003297790480378
		
		yield Xcoordinate, Ycoordinate, angle, azimuth
		
		if p > num:
			done = False
			yield done

def density(m, xdata, ydata, shapefile):
	'''This function converts the lat/lon of the input map to meters assuming an equirectangular projection.
	It then creates a grid at 100mpp, bins the input data into the grid (density) and creates a 
	histogram.  Finally, a mesh grid is created and the histogram is plotted in 2D over the basemap.
	
	If the shapefile flag is set to true a shapefile is created by calling the shapefile function.'''
	
	#Convert from DD to m to create a mesh grid.
	xmax = (m.xmax) / 0.003297790480378
	xmin = (m.xmin) / 0.003297790480378
	ymax = (m.ymax) / 0.003297790480378
	ymin = (m.ymin) / 0.003297790480378
	
	#Base 100mpp 
	nx = 1516
	ny = 2123
	
	#Convert to numpy arrays
	xdata = numpy.asarray(xdata)
	ydata = numpy.asarray(ydata)
	
	#Bin the data & calculate the density
	lon_bins = numpy.linspace(xdata.min(), xdata.max(), nx+1)
	lat_bins = numpy.linspace(ydata.min(), ydata.max(), ny+1)
	density, _, _ = numpy.histogram2d(ydata, xdata, [lat_bins, lon_bins])

	#If the user wants a shapefile, pass the numpy arrays
	if shapefile == True:
		print "Writing model output to a shapefile."
		create_shapefile(xdata, ydata)
		
	#Create a grid of equally spaced polygons
	lon_bins_2d, lat_bins_2d = numpy.meshgrid(lon_bins, lat_bins)
	if density.max() <= 5:
		maxden = 10
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
	parser.add_argument('velocity', action='store',type=int, help='The veloctiy of particles ejected. Typically around 300.')
	parser.add_argument('-i', '--iterations', action='store', type=int, dest='i',default=500, help='The number of ejection iterations to perform.')
	parser.add_argument('--shapefile', action='store_true', default=False, dest='shapefile', help='Use this flag to generate a shapefile, in Moon_2000GCS, of the point data.')
	parser.add_argument('--fast', action='store_true', default=False, dest='multi', help='Use this flag to forgo creating a visualization and just create a shapefile.  This uses all available processing cores and is substantially faster.')
	args = parser.parse_args()
	
	#Assign the user variables to the globals, not great form, but it works.
	velocity = args.velocity
	num = args.i
	
	#If the user wants to process quickly then we omit the visualization and multiprocess to generate a shapefile
	if args.multi == True:
		import multiprocessing
		cores = multiprocessing.cpu_count()
		cores *= 2
		step = num // cores
		xarray = numpy.frombuffer(multiprocessing.RawArray(ctypes.c_double, num))
		yarray = numpy.frombuffer(multiprocessing.RawArray(ctypes.c_double, num))
		init(xarray,yarray)
		jobs = []
		for i in range(0, num, step):
			p = multiprocessing.Process(target=strom_multi, args=(xarr,yarr,slice(i, i+step)), )
			jobs.append(p)
		for job in jobs:
			job.start()
		for job in jobs:
			job.join()
		#Write out a shapefile	
		create_shapefile(xarr, yarr)
	

	else:
		#Visualization - setup the plot
		fig = plt.figure(figsize=(15,10))
		ax1 = fig.add_subplot(1,2,1)
		pt, = ax1.plot([], [],'ro')
		xdata, ydata = [], []
	
		#Map
		lon_min = -101.5
		lon_max = -94.5
		lat_min = -32.5
		lat_max = -27.5
	
		m = Basemap(projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,
		    llcrnrlon=lon_min,urcrnrlon=lon_max,resolution=None, rsphere=(1737400.0,1737400.0))
		m.drawmeridians(numpy.arange(lon_min, lon_max+1, 1), labels=[0,0,0,1])
		m.drawparallels(numpy.arange(lat_min,lat_max+1, 0.5), labels=[1,0,0,0])
		
		#Read the input image
		im = imread('wac_clipped2.png')
		m.imshow(im, origin='upper', cmap=cm.Greys_r)
	
		def run(data):
			if data == False:
				density(m2,xdata, ydata, args.shapefile)
			else:
				x,y, angle, azimuth = data
				xorigin, yorigin = (-97.7328, -30.0906, ) #This is an estimate
				x += xorigin
				y += yorigin
				#Center the ejection origin at the pixel origin
				#Plot the data
				xdata.append(x)
				ydata.append(y)
				
				pt.set_data(xdata, ydata)
				print 'Angle: %f, Azimuth: %f, xCoordinate: %f, yCoordinate: %f' %(angle, azimuth,x,y)
				return pt,
		
		#Plot the volcano in the middle...
		xpt, ypt = m(-97.7328, -30.0906, ) #This is an estimate
		plt.plot(xpt, ypt, 'b^', markersize=10)
		#Run the animation
		ani = animation.FuncAnimation(fig, run,stromboli, interval=1, repeat=False, blit=False)
	
		plt.title('Interactive Deposition')
		ax2 = fig.add_subplot(1,2,2)
		ax2.set_title('Impacts / 1000m')
		
		#Map
		lon_min = -101.5
		lon_max = -94.5
		lat_min = -32.5
		lat_max = -27.5
	
		m2 = Basemap(projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,
		    llcrnrlon=lon_min,urcrnrlon=lon_max,resolution=None, rsphere=(1737400.0,1737400.0))
		m2.drawmeridians(numpy.arange(lon_min, lon_max+1, 1), labels=[0,0,0,1])
		m2.drawparallels(numpy.arange(lat_min,lat_max+1, 0.5), labels=[1,0,0,0])	
		
		m2.imshow(im, origin='upper', cmap=cm.Greys_r)
		
		#Plot the volcano in the middle...
		xpt, ypt = m2(-97.7328, -30.0906, ) #This is an estimate
		plt.plot(xpt, ypt, 'r^', markersize=10)
		
		plt.show()
		
		#Save the animation 
		#ani.save('simulation.mp4', fps=10)
	
	


	

	