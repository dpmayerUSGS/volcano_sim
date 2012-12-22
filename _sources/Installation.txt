.. _installation:

*********************************************
Installing the Strombolian Volcano Simulator
*********************************************

.. toctree::
    :maxdepth: 2
    
SVS Dependencies
====================

SVS requires Python and depends on several other freely available Python
modules. Prior to installing SVS, you should make sure its dependencies are met.

.. list-table:: SVS Dependencies
   :header-rows: 1
   :widths: 20, 25
   :class: center

   * - Dependency
     - Requirement
   * - `Python 2.7+ <http://www.python.org>`_
     - Required
   * - `NumPy <http://numpy.scipy.org/>`_
     - Required
   * - `GDAL <http://gdal.org/>`_
     - Required
   * - `GDAL Python Bindings <http://pypi.python.org/pypi/GDAL/>`_ 
     - Required
   * - `Matplotlib <http://matplotlib.org/users/installing.html>`_ 
     - Required
   * - `Python Imaging Library <https://developers.google.com/appengine/docs/python/images/installingPIL>`_
     - Required
   * - `Basemap <http://matplotlib.org/basemap/users/installing.html>`_
     - Required
     
Note that SVS is not tested with Python 3.x.

Downloading
============
SVS is hosted on github.  If you plan on using, but not contributing to the software, simply click this `link to download <https://github.com/jlaura/volcano_sim/archive/master.zip>`_ a .zip file containing SVS.

Conversely, if you are a git user and plan to contribute to the project (via a branch), use git clone.::

   $ git clone git://github.com/jlaura/volcano_sim.git
   
   This is a read only view.  You will need to branch the project to make changes.

Installing 
==========
SVS is distributed as a stand alone script and therefore does not require installation.  Simply place the script in a convenient directory.

Note that SVS does have a number of dependencies.  These facilitate data processing, shapefile output generation, and topographic profile extraction.  They are required and should therefore be installed prior to attempting to run SVS.

Installation on OS X
---------------------
Python
++++++
	Python ships with Mac OS X.  It is not necessary to install a different version.  Should you wish to, numerous online tutorials cover the installation of an additional python installation.  As of 10.6 (possibly earlier) the default Python installation should be 64bit.

Numpy
+++++
	
	Numerical Python is available for OS X via either pip or easy_install::
	
	   $ easy_install numpy
	   $ pip install numpy
	   
NumPy, PIL, MatPlotLib
+++++++++++++++++++++++

	Alternatively, you can install NumPy, along with PIL and Matplotlib via binaries.  These are compiled and made available by KyngChaos via his 
	
	`OS X GIS Ports <http://www.kyngchaos.com/software/python>`_
	
	Simply download the binares and install them.  While you are there, you might grab SciPy, it will be useful sometime soon on some other project! 

GDAL & GDAL Python Bindings
++++++++++++++++++++++++++++

	If you are using a package manager (Fink, MacPorts, Homebrew) install GDAL and the python bindings via that.  Otherwise, KyngChaos provides precompiled binaries for installation.  
	
	`GDAL Complete <http://www.kyngchaos.com/software/frameworks>`_
	
Basemap
+++++++

	This is slightly more complex and an OS X DMG will be forth coming.  The simplest methods, installation via automated source compilation is documented by the `NASA Modelling Guru <https://modelingguru.nasa.gov/docs/DOC-1847>`_.  In short, if you have macports, fink, or homebrew installed you can utilize one of the following, respectively.  Otherwise, you are going to need to build via source...::
	
	$ fink install matplotlib
	$ port install py-matplotlib-basemap
	$ brew install basemap
	
Installation on Windows
------------------------
Windows installation

NumPy, PIL, Basemap, Matplotlib, GDAL Python Bindings
++++++++++++++++++++++++++++++++++++++++++++++++++++++
	Christopher Gohlke has made a large number of binaries available for windows users.  All dependencies save the core GDAL libraries can be installed via his site.

	`Windows Python Packages <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_
	
.. warning::
	Install the GDAL core package before installing the gdal python bindings.
	
GDAL Core
++++++++++
	Installation of GDAL Core is slightly more complex.  First, download the binary package from `GIS Internals <http://www.gisinternals.com/sdk/PackageList.aspx?file=release-1600-gdal-1-9-2-mapserver-6-2-0.zip>`_.  This is gdal-##-####-core.msi.  If you are an ArcGIS user, you likely want the MSVC 2008 version.  Install this package as you normally would.

	Two tutorials will be of assistance in getting GDAL setup in your PATH.  Either tutorial covers the installation process.
	
	1. `My tutorial will exist as long as my Penn State account stays active <http://php.scripts.psu.edu/jzl5325/wordpress/?p=60>`_
	2. This `USU tutorial <http://www.gis.usu.edu/~chrisg/python/2009/docs/gdal_win.pdf>`_ also covers installation as a pdf.
	 

Installation on Linux
----------------------
	The simplest installation for users who are likely comfortable with more complex installations!::
	
	$ sudo apt-get install python-pip 
	$ sudo apt-get install gdal-bin python-gdal 
	$ pip install matplotlib 
	$ pip install numpy 
	$ pip install basemap 

.. note::
   If for some strange reason python is not already installed, you will be asked to install python running the first command, above.  