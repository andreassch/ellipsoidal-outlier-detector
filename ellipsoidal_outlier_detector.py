
"""
# Ellipsoidal outlier detection module

This module can be used to do ellipsoidal outlier detection in Python.  Given a scatter plot of `M` points in `ndim`-dimensions, you can select the points that are farthest from the center.  

** Basic usage **

Import and use the following functions : 
- get_outliers: Does one ellipsoidal outlier detection step.
- get_total_partition: Makes a sequence of calls to `get_outliers`, returns lists. 
- get_filter_index: Helper routine for applying a filter to new data. 

** Details ** 

The `EllipsoidSolver` class is used to actually do the optimization that finds the minimum volume ellipsoid that contains all the specified points.  The optimization depends on the CVXOPT package.  For CVXOPT documentation see the website, cvxopt.org.  Also, the book "Convex Optimization" by [Boyd and Vandenberghe] might be useful.  

** Notes **
There may be some room to improve performance using sparse arrays.  But it should work fairly well up to ndim = 8 and M = 10,000. [July 2017]


"""




# Future
from __future__ import print_function

# numpy and scipy
import numpy as np
import numpy.random as random
import scipy as sp
import scipy.linalg as la


# Convex optimization
# install cvxopt if needed
# if you have anaconda: "conda install cvxopt"
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from cvxopt import blas, lapack, sqrt, mul, cos, sin, log

## Constants
EPS = sp.finfo(float).eps





## Class definition for the El
class EllipsoidSolver(object):
    """
    Class for constructing a function to pass to the convex 
    problem solver of CVXOPT.  

    """

    def __init__(self, xarray, a0=None):
        """
        Construct an interface to general convex problem solver
        of CVXOPT: 'cvxopt.solver.cp'.

        in
        --
        xarray - scattered filter vector array (M, ndim)
        a0     - initial guess for the state vector 
        
        out
        ---
        F      - callable function (see docs for cvxopt.solver.cp)

        """

        ## Store the samples
        self.xarray = xarray
        self.M, self.ndim = xarray.shape[:]
        self.Na = int(self.ndim * (self.ndim + 1) / 2.0 + self.ndim)

        ## Get arrays for unpacking the vectorized ellipse parameter vector a0
        self.D, self.B, self.Nindex_ij, self.IJindex_n = (
            self._get_D_B_Nindex_IJindex(self.ndim, self.Na))

        
        ## Define the initial guess
        if a0 is None:

            # Compute the mean and maximum deviation
            xmean = self.xarray.mean(0)
            maxdev = la.norm(self.xarray - xmean[None, :], axis=1).max()

            ## Make a sphere containing the points
            b0 = sp.zeros(self.ndim)
            b0[:] = - xmean / maxdev           # shift vector
            A0 = sp.eye(self.ndim) / maxdev    # circle

            # Define the starting vector a0
            a0 = sp.zeros(self.Na)
            a0[-self.ndim:] = b0
            a0[:-self.ndim] = A0.ravel()[self.Nindex_ij]
    
        # Store the initial condition
        self.a0 = a0
        self.A0 = self.get_A(a0)
        self.b0 = self.get_b(a0)
        
    # Make the instances callable to interface with CVXOPT
    def __call__(self, a=None, z=None):
        """
        Loewner-John ellipsoid
        
        minimize     log det A^-1
        subject to   || A(a) x_n + b(a) ||_2^2  - 1.0 <= 0  for  n=1,...,m
        
        """


        # Convert input to numpy arrays
        n = self.Na
        m = self.M
        
        ## Handle the case for determining the initial value
        if a is not None:
            a = sp.array(a).reshape(n)
        else:
            return m, matrix(self.a0)
    
    
        ## Calculations 
        if z is not None: 
            z = sp.array(z).reshape(m + 1)
        
            ## Compute the full outputHessian
            fout = sp.zeros([m+1])
            dfout = sp.zeros([m+1, n])
            ddfout = sp.zeros([n, n])

            fout[0] = self.get_f0(a)
            dfout[0] = self.grad_f0(a)
            ddfout[:, :] += z[0] * self.hess_f0(a)
        
            fout[1:] = self.get_f1(a)
            dfout[1:, :] = self.grad_f1(a)
            
            ddfout[:, :] += self.hess_f1(a, z) #(z[1:, None, None] * self.hess_f1(a)).sum(0)
            # ddfout[:, :] += (z[1:, None, None] * self.hess_f1(a)).sum(0)
         
            # Return the full output
            return (matrix(fout.reshape(m+1, 1)), 
                    matrix(dfout), 
                    matrix(ddfout))
        
        else:
        
            # Check the domain
            A = self.get_A(a)
            lams = la.eigvalsh(A)
            if lams.min() <= -10 * EPS:
                return (None, None)
            
            ## Compute partial output without hessian
            fout = sp.zeros([m+1])
            dfout = sp.zeros([m+1, n])
            
            fout[0] = self.get_f0(a)
            dfout[0] = self.grad_f0(a)
        
            fout[1:] = self.get_f1(a)
            dfout[1:, :] = self.grad_f1(a)

            return (matrix(fout.reshape(m+1, 1)), 
                    matrix(dfout))
  
    # Define a call to the solver
    def get_optimal_ellipse(self, solvers=solvers):
        """
        Call the solver to fine the minimal volume ellipsoid.
        """
        # Solver the minimum volume ellipse problem
        sol = solvers.cp(self)
        avec = np.array(sol['x']).flatten()

        # Compute the standard ellipse parameters
        b = self.get_b(avec)
        A = self.get_A(avec)
        vol = 1.0 / la.det(A)

        return avec, vol, A, b

    def get_xinxout(self,A, b, EPS=EPS):
        """
        Split the filter array into interior an boundary points. 

        in 
        --
        A     - Ellipse array 
        b     - Ellipse offset vector
        
        out 
        ---
        xin   - Part of xarray that is in ellipse
        xout  - Part of xarray that is outside the ellipse
        Iin   - Bolean index array for points in ellipse
        """
        x = self.xarray
        v = sp.dot(A, x.T) + b[:, None]
        Iin = (v**2).sum(0)-1 <  - 10 * sp.sqrt(EPS)
        Iout = (v**2).sum(0)-1 >= - 10 * sp.sqrt(EPS)
        xin = x[Iin, :]
        xout = x[Iout, :]

        return xin, xout, Iin 

    
    ## Get arrays for parameterizing the ellipse
    def _get_D_B_Nindex_IJindex(self, ndim, Na):
        """
        Define the array Dnij so that, 
        Aij = SUM_n Dnij * an. 
        
        Define the array Bni so that, 
        bi = SUM_n Bni * an.
        
        *Notes*
        The array Dnij is pretty big and I don't yet take
        advantage of sparcity so it is suboptimal. 
    
        """
    
        # Zero array of correct size
        D = sp.zeros((Na, ndim, ndim))
        Nindex_ij = sp.zeros((Na-ndim,), dtype=int)
        IJindex_n = sp.zeros((ndim**2), dtype=int)

        
        # Loop over parameters for ellipse
        n=0
        for i in range(ndim):
            for j in range(0, i+1):
                # Define the array to project a0 to A
                D[n, i, j] = 1.0
                D[n, j, i] = 1.0

                # Define the index arrays to get 
                Nindex_ij[n] = j + i * ndim   # a[:-ndim] = A.ravel()[Nindex_ij]
                IJindex_n[j + i * ndim] = n
                IJindex_n[i + j * ndim] = n   # A.ravel()[:] = a[IJindex_n]
                n+=1
                    
        # initialize B
        B = sp.zeros((Na, ndim))
        B[-ndim:, :] = sp.eye(ndim)
        
        return D, B, Nindex_ij, IJindex_n

    ## Get the ellipse array from Functions
    def get_A(self, a):
        """
        Compute the ellipse array A.
        
        The ellipse is,
          ||dot(A, x) + b||^{2} <= 1. 
        for array A and centroid vector -b.  
        """
        return sp.tensordot(self.D, a, axes=[(0,), (0,)])
    
    def get_b(self, a):
        """
        Compute the ellipse centroid -b.
        
        The ellipse is,
          ||dot(A, x) + b||^{2} <= 1. 
        for array A and centroid vector -b.  
        """
        return sp.tensordot(self.B, a, axes=[(0,), (0,)])



    # Define cost functions, constraint functions, and derivatives for cvxopt
    
    ## Cost function: 
    def get_f0(self, a):
        "Compute the objective function: -log(det(A(a)))."
        A = self.get_A(a)
        Ainv = la.inv(A)
        out = sp.log(la.det(Ainv))
        return out
    
    def grad_f0(self, a):
        "Compute the gradient of the objective function: -log(det(A(a)))."
        A = self.get_A(a)
        Ainv = la.inv(A)
        E = sp.dot(Ainv, self.D).transpose(1,0,2)
        out = -sp.trace(E, axis1=1, axis2=2)
        return out

    def hess_f0(self, a):
        "Compute the hessian of the objective function: -log(det(A(a)))."
        A = self.get_A(a)
        Ainv = la.inv(A)
        E = sp.dot(Ainv, self.D).transpose(1,0,2)
        EE = sp.dot(E, E).trace(axis1=1, axis2=3)
        H = (1.0 / 2.0) * (EE + EE.T)
        return H

    ## The constraint function
    def get_f1(self, a):
        "Define the nth constraint function."
        b = self.get_b(a)
        A = self.get_A(a)
        Ax = sp.dot(A, self.xarray.T)
        val = ((b[:, None] + Ax[:, :])**2).sum(0) - 1.0
        return val

    def grad_f1_slow(self, a):
        "Define the gradient for each convex inequality."
        b = self.get_b(a)
        A = self.get_A(a)
        Ax = sp.dot(A, self.xarray.T)
        vec0 = b[:, None] + Ax[:, :]                                #  in
        vec1 = self.B[:, :, None] + sp.dot(self.D, self.xarray.T)[:, :, :]    # kin
        vec_isum = 2.0 * (vec0[None, :, :] * vec1[:, :, :]).sum(1).transpose()
        return vec_isum

    def grad_f1(self, a):
        "Define the gradient for each convex inequality."

        # Initialize the output vector
        out = sp.zeros((self.M, self.Na))

        # Preliminary calculation
        _xx = sp.einsum('mi,mj->mij', self.xarray, self.xarray)

        # Compute the four terms
        _Da = sp.tensordot(self.D, a, axes=[(0,), (0,)])
        _DDa = sp.tensordot(self.D, _Da, axes=[(1,), (0,)])
        xxDDa = sp.tensordot(_xx.reshape(self.M, self.ndim**2),
                             _DDa.reshape(self.Na, self.ndim**2),
                             axes=[(-1,), (-1,)])

        _BDa = sp.dot(self.B, _Da)
        xBDa = sp.inner(self.xarray, _BDa)

        _Ba = sp.dot(a, self.B)
        _DBa = sp.dot(_Ba, self.D)
        xDBa = sp.tensordot(self.xarray,
                            _DBa, axes=[(-1,), (-1,)])

        BBa = sp.dot(self.B, _Ba)
        
        # compute the gradient by summing the four terms
        out[:, :] = 2.0 * (xxDDa + xBDa + xDBa + BBa)

        return out

    def hess_f1_slow(self, a, z):
        "Define the hessians for each convex inequality."
        vec1 = self.B[:, :, None] + sp.dot(self.D, self.xarray.T)[:, :, :]    # kin -> nkk'
        vec_kkn = 2.0 * (vec1[:, None, :, :] * vec1[None, :, :, :]).sum(2)
        vec_nkk = vec_kkn.transpose(2,0,1)

        ## Update with z
        out = (z[1:, None, None] * vec_nkk).sum(0)
        return out

    def hess_f1(self, a, z):
        "Define the hessian for the convex inequalities."

        ## Preliminaries
        z = z[1:]
        _z1 = z.sum()
        _zx = sp.dot(z, self.xarray)
        _zxx = (z[:, None, None] *
                self.xarray[:, :, None] *
                self.xarray[:, None, :]).sum(0)
        
        # Initialize the output "Hessian" array
        out = sp.zeros((self.Na, self.Na))

        ## There are four terms to compute
        _aux0 = _z1 * sp.tensordot(self.B, self.B, axes=[(-1,), (-1,)])
        _Dzx = sp.dot(self.D, _zx)
        _aux1 = sp.tensordot(self.B, _Dzx, axes=[(-1,), (-1,)])
        _aux2 = sp.tensordot(_Dzx, self.B, axes=[(-1,), (-1,)])
        _Dzxx = sp.dot(self.D, _zxx)
        _aux3 = sp.tensordot(_Dzxx, self.D, axes=[(1,2), (1,2)])

        ## output array 
        out[:, :] = 2.0 * (_aux0 + _aux1 + _aux2 + _aux3)

        return out
        
# Functions for easy interface to outlier computing outliers easily
#

## Do a single outlier detection step
def get_outliers(xarray, avec0=None, EllipsoidSolver=EllipsoidSolver):
    """
    Compute the outliers and split them off.  This routine finds the minimum 
    volume ellipsoid that contains all points.  Finds parameters for a 
    single filter step. 

    in
    --
    xarray   - array shaped (M, d) for M samples of d-dimensional vecs
    avec0    - Optional initial guess for the ellipse state vector 


    out
    ---
    xin      - samples inside the minimum ellipse 
    xout     - samples on the boundary of the ellipse
    bol_Iin  - index of samples i: 0 <= i < M 
    vol      - constand proportional to the volume (1.0 / det(A))
    A        - Array defining the ellipse 
    b        - Vector defining the offset of ellipse 
    avec     - state vector for minimum volume ellipse 
    
    
    * Note *
    Use the kwarg avec0 to initialize the ellipse optimization.  


    """
    
    # Define a solver for these filter vectors
    esolver = EllipsoidSolver(xarray, a0=avec0)
    
    # Get the min. vol. ellipse containing all points 
    avec, vol, A, b = esolver.get_optimal_ellipse()
    
    # Split the xarray into interior and boundary points
    xin, xout, bool_Iin = esolver.get_xinxout(A, b)

    return xin, xout, bool_Iin, vol, A, b, avec


## Do a repeated outlier detection until a fraction of points are removed 
def get_total_partition(xarray, alpha=0.5, EllipsoidSolver=EllipsoidSolver):
    """
    Compute outliers in sequance for all points.
    
    
    in
    --
    xarray      - array shaped (M, d) for M samples of d-dimensional vecs
    alpha       - stop partition when xin is smaller than (M*alpha, d)

    out
    ---
    bool_Iin_list    - list of bool index arrays for all inliers each is ~(M,)
    bool_Iout_list   - list of bool index arrays for the latest outliers ~(M,)
    vols             - array of ellipsoid volumes
    As               - A_arrs define the covarience of ellipse
    bs               - b_vecs define the center of the ellipse
    

    Usage
    ------
    The boolean arrays that are returned are used as follows:

      xin = xarray[bool_Iin, :]
      xout= xarray[bool_Iout, :]

    This xout contains the outliers for the latest step only. 

    """
    
    # Get problem dimensions
    M, d = xarray.shape[:]

    # Make a index array to keep track of index_in
    Iin = sp.arange(M)
    bool_Iin = sp.ones((M,), dtype=bool)

    # Initialize output
    bool_Iin_list = []
    bool_Iout_list = []
    vols = sp.zeros((M,))
    As = sp.zeros((M, d, d))
    bs = sp.zeros((M, d))
    
    # Initialize 
    avec0 = None
    xin = xarray
    
    # Loop until the set of points has been partitioned
    for i in range(M):
        
        # Get the next ellipse
        xin, xout, bool_Iin_next, vol, A, b, avec0 = get_outliers(xin, avec0=avec0)


        ## Update the index arrays Iin and bool_Iin
        Iout_next = Iin[~bool_Iin_next]
        bool_Iin[Iout_next] = False
        Iin = Iin[bool_Iin_next]

        ## Define a bool array indicated most recent outliers
        bool_Iout_next = sp.zeros(M, dtype=bool)
        bool_Iout_next[Iout_next] = True

        
        # Store output
        bool_Iin_list.append(bool_Iin.copy())
        bool_Iout_list.append(bool_Iout_next)
        vols[i] = vol
        As[i, :, :] = A
        bs[i, :] = b

                
        if xin.shape[0] < M * alpha:
            ibound = i + 1            
            break
    
        
    # Prepare output
    vols = vols[:ibound]
    As = As[:ibound, :, :]
    bs = bs[:ibound, :]
    
    return bool_Iin_list, bool_Iout_list, vols, As, bs


## Define a function for filtering points with a given ellipse
def get_filter_index(xarray_new, A, b):
    """
    Return an index array for selecting the point which are inside
    the ellipse defined by (A, b).
    
    in 
    --
    xarray_new    - (M', ndim)
    A             - (ndim, ndim)
    b             - (ndim,)
    
    out
    ---
    bool_In       - (M',) Index array for selecing points in the ellipse

    
    * usage *
    ---------
    xarray[bool_Iin, :]     --->    gives the points inside the ellipse 


    """

    # Get the points in the ellipse
    v = sp.dot(xarray_new, A) + b     # v~(M', ndim)
    bool_Iin_new = sp.einsum('ij,ij->i', v, v) <= 1.0

    # Return the boolean index.  USAGE: `xarray_new[bool_Iin_new, :]`
    return bool_Iin_new

    


# Main program
if __name__ == "__main__":


    # Imports for plotting the scatter plots and outliers
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ## Make a plot for the case where ndim is 3
    def plot_xinxout_3d(xin, xout, A, b):
        """
        Plotting routine for the case where the number of filter 
        parameters is 3.  
        
        """

        Ainv = la.inv(A)
    
        # Make a 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.scatter(xin[:,0], xin[:,1], xin[:,2], marker='.')
        ax.scatter(xout[:,0], xout[:,1], xout[:,2], marker='o', c='r')

    
        ## Get points on a sphere
        thetas = sp.linspace(100*EPS, (1.0-100*EPS) * sp.pi, 20)
        phis = sp.linspace(0.0, 2*sp.pi, 20)
        TT, PP = sp.meshgrid(thetas, phis, indexing='ij')

        xx = sp.cos(PP) * sp.sin(TT)
        yy = sp.sin(PP) * sp.sin(TT)
        zz = sp.cos(TT)


        Ainv_xx = Ainv[:,0, None, None] * (xx - b[None, None, 0])
        Ainv_yy = Ainv[:,1, None, None] * (yy - b[None, None, 1])
        Ainv_zz = Ainv[:,2, None, None] * (zz - b[None, None, 2])

        xyz_ellipse = Ainv_xx + Ainv_yy + Ainv_zz
        xe = xyz_ellipse[0]
        ye = xyz_ellipse[1]
        ze = xyz_ellipse[2]

    
        ax.plot_wireframe(xe, ye, ze, linewidth=0.5)
    
    
        return fig


    
    # Generate random samples for testing the algorithm
    Ndim = 3
    Nsamples = 1000


    # Generate random points . . . some Gausian some Laplacian 
    mean = np.array(Ndim * [5.0,])
    cov = sp.eye(Ndim) + 5.0

    xarray = random.multivariate_normal(mean, cov, Nsamples)
    xarray[:50] = random.laplace(5, scale =10.0000001, size=(50, mean.size))


    # Compute the first partition
    xin, xout, bool_Iin, vol, A, b, avec = get_outliers(xarray)
    
    fig = plot_xinxout_3d(xin, xout, A, b)
    fig.show()


    
    # # Make a 3D figure
    # fig = plt.figure(0)
    # fig.clear()
    # ax = fig.add_subplot(1,1,1, projection='3d')
    # ax.scatter(xarray[:,0], xarray[:,1], xarray[:,2], marker='.')
    # fig.show()
