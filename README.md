

# Ellipsoidal outlier detection in Python

This module can be used to do ellipsoidal outlier detection in Python.  Given a scatter plot of `M` points in `ndim`-dimensions, you can select the points that are farthest from the center.

## Dependencies

You will need to have the Python package CVXOPT installed, and also `numpy` and `scipy`.  I recommend using the anaconda distribution of python and installing cvxopt with their package manager `conda` as follows: 

  $ conda install cvxopt


## Basic usage

Import and use the following functions : 
- get_outliers: Does one ellipsoidal outlier detection step.
- get_total_partition: Makes a sequence of calls to `get_outliers`, returns lists. 
- get_filter_index: Helper routine for applying a filter to new data. 

## Details 

The `EllipsoidSolver` class is used to actually do the optimization that finds the minimum volume ellipsoid that contains all the specified points.  The optimization depends on the CVXOPT package.  For CVXOPT documentation see the website, cvxopt.org.  Also, the book "Convex Optimization" by [Boyd and Vandenberghe] might be useful.  

## Notes 
There may be some room to improve performance using sparse arrays.  But it should work fairly well up to ndim = 8 and M = 10,000. [July 2017]

