import numpy as np
from scipy.linalg import lstsq

def rectangular_kernel(x):
	if np.abs(x) < 1:
		return 0.5
	else:
		return 0.0

def hammersteinApproximation( inputsignal, outputsignal, 
	kernel=rectangular_kernel, h=lambda n: n**(-0.25), 
	npoints=None, shift=True):
	K = np.vectorize(kernel)
	if not npoints:
		npoints = inputsignal.size
	hn = h(npoints)

	# TODO: Change to the statistically significant region
	approxPoints = np.linspace(np.min(inputsignal), np.max(inputsignal), npoints)
	mu = np.zeros(npoints)
	for i in range(npoints):
		numer = np.sum(outputsignal * K( (approxPoints[i]-inputsignal)/hn ))
		denom = np.sum(K( (approxPoints[i]-inputsignal)/hn ))
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
	K = np.vectorize(kernel)
	if not npoints:
		npoints = inputsignal.size
	hn = h(npoints)

	approxPoints = np.linspace(np.min(outputsignal), np.max(outputsignal), npoints)
	nu = np.zeros(npoints)
	for i in range(npoints):
		numer = np.sum(inputsignal * K( (approxPoints[i]-outputsignal)/hn ))
		denom = np.sum(K((approxPoints[i]-outputsignal)/hn))
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