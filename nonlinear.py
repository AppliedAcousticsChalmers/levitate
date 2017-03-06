import numpy as np
from scipy.linalg import lstsq
from scipy.special import legendre

def rectangular_kernel(calcP,measP,param):
	return np.where(np.abs(calcP-measP)<param,0.5,0.0)

def legendre_kernel(calcP,measP,param):
	calcP = np.asarray(calcP)
	measP = np.asarray(measP)
	pn = legendre(param)
	pn1 = legendre(param+1)
	with np.errstate(divide='ignore',invalid='ignore'):
		# This supresses the potential warning about dividing with zero.
		# `np.where` will handle the issue and choose the valid approach, 
		# but both expressions will be evaluated.
		return np.where( calcP==measP,
			(param+1.)/2 * (pn1.deriv()(calcP) * pn(calcP) - pn.deriv()(calcP)*pn1(calcP) ), 
			(param+1.)/2 * (pn(calcP)*pn1(measP) - pn(measP)*pn1(calcP))/(measP-calcP))
	# TODO: Will this be much faster if the second expression is evaluated everywhere first,
	# and the nan replaced by the correct values?
	# This could be implemented using `if any(np.isnan(out)):` and then replacing the nans.

class hammersteinModel:
	'''
	A Hammerstein Model Class

	Wraps all the functionality to create simulations of Hammerstein Models,
	calculate approximations for the linear and nonlinear parts from sampled
	input/output relations.
	'''

	def __init__(self):
		# TODO: Do stuff here!
		pass

	def __call__(self,inputsignal):
		'''
		Feeds an input signal through the hammerstein model.
		'''
		pass

	def approximateNonlinearity(self,inputsignal,outputsignal):
		pass

	def approximateLinearity(self,inputsignal,outputsignal):
		pass

	def setKernel(self,kernel):
		''' Selects kernel.
		
		Call with a string to select a kernel from the default kernels.
		This will reset the kernelParam, so any custom kernel parameters 
		must be set again.
		To use a custom kernel, change `model.kernel` directly, or 
		pass a callable for more safe changes.

		'''
		#TODO: Document better!
		if isinstance(kernel,str):
			if kernel.lower()[:4] == 'rect':
				self.kernel = rectangular_kernel
				self.kernelParam = lambda n: n**(-0.25)
			elif kernel.lower()[:4] == 'lege':
				self.kernel = legendre_kernel
				self.kernelParam = lambda n: np.ceil(n**0.25).astype('int')
			else:
				raise KeyError('Kernel `{}` is not an implemented default kernel!'.format(kernel))
		elif callable(kernel):
			self.kernel = kernel
		else:
			raise TypeError('`kernel` must be a string or a callable!')




def hammersteinApproximation( inputsignal, outputsignal, 
	kernel=rectangular_kernel, kernelParam=None, 
	npoints=None, shift=True):
	"""
	Return a approximation to the nonlinearity of a Hammerstein system.

	Parameters
	----------
	inputsignal : array_like
		The sampled input values to the system
	outputsignal : array_like
		The sampled output values from the system
	kernel : callable or string
		This will specify which kernel to use in the calculations.
		Can be a function with signature `kernel(calcPoints,measPoints,param)`
		that will act as the kernel for the approximation. 
		`calcP` is the value where the approximation is calculated, 
		`measP` is the sampled points, and `param` can be used to specify 
		additional parameters using the `kernelParam` parameter.
		Note that the kernel function must be vectorized for `measPoint`.
		Alternatively, this can be a string that specifies any of the following kernels:
			- rectangular : A rectangular kernel
			- legendre : A kernel representing series expansion in legendre polynomials
	kernelParam : callable or any
		Specifies parameters for the kernel.
		If this is callable it will be called with `npoints` and passed to the kernel.
		Otherwise it will be passed to the kernel as is. Make sure that any non-standard
		kernels have a matching kernelParam or does not make use of the param.
		This is per default used to specify the resonution parameter h(n) and order parameter N(n)
	npoints : int
		Use this to specify the number of points where the nonlinearity will be approximated.
		If set to `None` this will be set to the length of the input signal.
	shift : bool
		Specifies if shifting is applied. If `True` this will shift the approximation
		so that mu(0)=0.

	Returns
	-------
	approxPoints : ndarray
		This is the points where the approxiation is calculated.
	mu : ndarray
		This is the approximated nonlinearity. Will be `None` at points where
		the approximation of not valid.

	"""
	if isinstance(kernel, str):
		# Choose the correct kernel and kernelParam pair.
		if kernel.lower()[:4] == 'rect':
			kernel = rectangular_kernel
			if not kernelParam: kernelParam = lambda n: n**(-0.25)
		elif kernel.lower()[:4] == 'lege':
			kernel = legendre_kernel
			if not kernelParam: kernelParam = lambda n: np.ceil(n**0.25).astype('int')
			# TODO: Is this a good choise of order??
	if not npoints:
		npoints = inputsignal.size
	if not kernelParam:
		kernelParam = lambda n: n**(-0.25)
	if callable(kernelParam):
		kernelParam = kernelParam(npoints)

	# TODO: Change to the statistically significant region
	approxPoints = np.linspace(np.min(inputsignal), np.max(inputsignal), npoints)
	mu = np.zeros(npoints)
	for i in range(npoints):
		numer = np.sum(outputsignal * kernel(approxPoints[i],inputsignal,kernelParam) )
		denom = np.sum(kernel(approxPoints[i],inputsignal,kernelParam))
		if denom == 0:
			mu[i] = None
		else:
			mu[i] = numer/denom

	if shift:
		closezero = np.searchsorted(approxPoints,0)
		if -approxPoints[closezero-1] < approxPoints[closezero]:
			closezero -= 1
		mu = mu - mu[closezero]

	return approxPoints, mu


def hammersteinLinearApproximation( inputsignal, outputsignal, len=None, nonlinearFun=None):
	# TODO: Handle input lengths!
	if not nonlinearFun:
		nonlinearFun = lambda x: x
	nlfun = np.vectorize(nonlinearFun)
	if not len:
		len = 256 # TODO: How to choose this in a proper way?
	Y = outputsignal[len:]
	mu = np.zeros((Y.size,len))
	for row in range(Y.size):
		mu[row] = nlfun(inputsignal[row+len:row:-1])
	IR=lstsq(mu,Y)
	return IR[0]



def hammerseinLinearApproximationCorrelation( inputsignal, outputsignal, len=None) :
	if not len:
		len = 256 # TODO: How to choose this? 
	kappa = np.zeros(len)

	kappa[0] = 1/len * np.sum(inputsignal*outputsignal)
	for i in range(1,len):
		kappa[i] = 1/len * np.sum( inputsignal[:-i]*outputsignal[i:] )
	return kappa

def weinerApproximation( inputsignal, outputsignal, 
	kernel=rectangular_kernel, h=lambda n: n**(-0.25),
	npoints=None, shift=True):
	
	if not npoints:
		npoints = inputsignal.size
	hn = h(npoints)

	approxPoints = np.linspace(np.min(outputsignal), np.max(outputsignal), npoints)
	nu = np.zeros(npoints)
	for i in range(npoints):
		numer = np.sum(inputsignal * kernel(approxPoints[i],outputsignal,hn ))
		denom = np.sum(kernel(approxPoints[i],outputsignal,hn))
		if denom == 0:
			nu[i] = None
		else:
			nu[i] = numer/denom

	if shift:
		closezero = np.searchsorted(approxPoints,0)
		if -approxPoints[closezero-1] < approxPoints[closezero]:
			closezero -= 1
		nu = nu - nu[closezero]

	return approxPoints, nu