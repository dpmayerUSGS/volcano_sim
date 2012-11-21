from osgeo import gdal
import osr
import numpy

#Read the input DTM
ds = gdal.Open('wac_dtm_int16_clipped.tif')
dtm = ds.ReadAsArray()

geotransform = ds.GetGeoTransform()
dev = (geotransform[1]*geotransform[5] - geotransform[2]*geotransform[4])
gtinv = ( geotransform[0] , geotransform[5]/dev, - geotransform[2]/dev, geotransform[3], - geotransform[4]/dev, geotransform[1]/dev)

originx = -97.788
originy = -30.263
destx = -95.768829
desty = -30.979342
distance = 64966.9856545
xpt = numpy.linspace(originx,destx,num=distance/100, endpoint=True)
ypt = numpy.linspace(originy,desty,num=distance/100, endpoint=True)
xpt -= geotransform[0]
ypt -= geotransform[3]
xsam = numpy.round_((gtinv[1] *xpt + gtinv[2] * ypt), decimals=0)
ylin = numpy.round_((gtinv[4] *xpt + gtinv[5] * ypt), decimals=0)
xsam.astype(int); ylin.astype(int)
print xsam
exit()
newarr = dtm[ylin.reshape(-1),xsam.reshape(-1)]
print newarr
exit()
print len(newarr)
for x in range(len(newarr)):
    newarr[x] = dtm[ylin[x]][xsam[x]]
print newarr
