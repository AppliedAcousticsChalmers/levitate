import numpy as np
from scipy.linalg import lstsq
from scipy.special import legendre
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.fftpack import fft, ifft

# Normal kernels
def rectangular_kernel(calcP, measP, param):
    return np.where(np.abs(calcP - measP) < param, 0.5, 0.0)

def triangular_kernel(calcP, measP, param):
    return np.where(np.abs(calcP - measP) < param, 1-np.abs(calcP-measP) / param, 0)

def parabolic_kernel(calcP, measP, param):
    return np.where(np.abs(calcP - measP) < param, 3/4*(1 - ((calcP-measP) / param)**2), 0)

def gaussian_kernel(calcP, measP, param):
    # Also called Gauss-Weierstass kernel
    return np.exp(-0.5 * ((calcP - measP) / param)**2) / np.sqrt(2*np.pi)

def poisson_kernel(calcP, measP, param):
    return 1/(np.pi*(1+((calcP-measP)/param)**2))

def sincsquare_kernel(calcP,measP,param):
    # Also called Fejer kernel
    # TODO: What about the pi in the denominator inside sinc? Does it matter if it is there or not?
    # It seems as if this is only a scaling of the default parameter
    return np.sinc( (calcP-measP)/(np.pi*param) )**2/np.pi

def sinc_kernel(calcP,measP,param):
    return np.sinc( (calcP-measP)/(np.pi*param) )/np.pi

def lebesgue_kernel(calcP,measP,param):
    return np.exp(-np.abs(calcP-measP)/param)/2

#### Orthogonal polynomial kernels ####
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

    def __init__(self,kernel='rectangular',npoints=None,irlen=256,shift=True,normalize=True):
        self.setKernel(kernel)
        self.npoints = npoints
        self.shift = shift
        self.irlen = irlen
        self.normalize = normalize

    def __call__(self,inputsignal,trunkate=True):
        '''
        Feeds an input signal through the hammerstein model.
        '''
        outputsignal = fftconvolve(self.nonlinearity(inputsignal),self.impulseResponse,'full')
        if trunkate:
            return outputsignal[:inputsignal.size]
        else:
            return outputsignal

    def approximateNonlinearity(self,inputsignal,outputsignal):
        inputsignal = np.asarray(inputsignal)
        outputsignal = np.asarray(outputsignal)
        if self.npoints:
            npoints = int(self.npoints)
        else:
            npoints = inputsignal.size
        if callable(self.kernelParam):
            kernelParam = self.kernelParam(npoints)
            # TODO: The theory always wants to call this with the length of the input signal,
            # regardless of how dense the calculation will be.
        else:
            kernelParam = self.kernelParam

        if self.normalize:
            scale = np.max(inputsignal) - np.min(inputsignal)
            inputsignal = 2*(inputsignal-np.mean(inputsignal))/scale


        approxPoints = np.linspace(np.min(inputsignal), np.max(inputsignal), npoints)
        mu = np.zeros(npoints)
        for i in range(npoints):
            numer = np.sum(outputsignal * self.kernel(approxPoints[i],inputsignal,kernelParam) )
            denom = np.sum(self.kernel(approxPoints[i],inputsignal,kernelParam))
            if denom == 0:
                mu[i] = None
            else:
                mu[i] = numer/denom

        if self.shift:
            closezero = np.searchsorted(approxPoints,0)
            if -approxPoints[closezero-1] < approxPoints[closezero]:
                closezero -= 1
            mu = mu - mu[closezero]
        if self.normalize:
            scale = np.max(mu)-np.min(mu)
            mu/=scale

        self.rawApprox = approxPoints, mu
        finiteIdx = np.isfinite(mu)
        self.nonlinearity = interp1d(approxPoints[finiteIdx],mu[finiteIdx],bounds_error=False)
 
    def refineModel(self,inputsignal,outputsignal):
        #inputspectrum = fft(inputsignal)
        irpad = np.r_[self.impulseResponse,np.zeros(inputsignal.size-self.impulseResponse.size)]
        irspectrum = fft(irpad)
        outputspectrum = fft(outputsignal)
        nonlinearspectrum = outputspectrum/irspectrum
        nonlinearoutput = np.real(ifft(nonlinearspectrum))
        self.approximateNonlinearity(inputsignal,nonlinearoutput)
        self.approximateLinearity(inputsignal,outputsignal)


    def approximateLinearity(self, inputsignal, outputsignal):
        if hasattr(self,'nonlinearity'):
            # Approximate using the nonlinearity
            Y = outputsignal[self.irlen:]
            mu = np.zeros((Y.size,self.irlen))
            for row in range(Y.size):
                mu[row] = self.nonlinearity(inputsignal[row+self.irlen:row:-1])
            IR=lstsq(mu,Y)
            self.impulseResponse = IR[0]
            if self.normalize:
                self.impulseResponse /= np.max(np.abs(self.impulseResponse))
        else:
            raise NotImplementedError('Approximations of linearities before approximations of the nonlinearity is not implemented!')

    def setKernel(self, kernel, kernelParam=None):
        ''' Selects kernel.
        
        Call with a string to select a kernel from the default kernels.
        This will reset the kernelParam, so any custom kernel parameters 
        must be set again.
        To use a custom kernel, change `model.kernel` directly, or 
        pass a callable for more safe changes.

        '''
        #TODO: Document better!
        if isinstance(kernel, str):
            if kernel.lower()[:4] == 'rect':
                self.kernel = rectangular_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:3] == 'tri':
                self.kernel = triangular_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:4] == 'para':
                self.kernel = parabolic_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:5] == 'gauss':
                self.kernel = gaussian_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:4] == 'pois':
                self.kernel = poisson_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:6] == 'sincsq':
                self.kernel = sincsquare_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:4] == 'sinc':
                self.kernel = sinc_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:4] == 'lebe':
                self.kernel = lebesgue_kernel
                if not kernelParam: kernelParam = lambda n: n**(-0.25)
            elif kernel.lower()[:4] == 'lege':
                self.kernel = legendre_kernel
                if not kernelParam: kernelParam = lambda n: np.floor(n**0.25).astype('int')
            else:
                raise KeyError('Kernel `{}` is not an implemented default kernel!'.format(kernel))
        elif callable(kernel):
            self.kernel = kernel
        else:
            raise TypeError('`kernel` must be a string or a callable!')
        self.kernelParam = kernelParam

    def setOrthogonalBasis(self,basis,basisParam=None):
        # TODO: Docuent!
        if isinstance(basis,str):
            if basis.lower()[:4] == 'lege':
                self.orthogonalBasis = lambda n: ((2*n+1)/2)**0.5*legendre(n)
                if not basisParam: basisParam = lambda n: np.floor(n**0.25).astype('int')
            else:
                raise KeyError('Basis function `{}` is not an implemented default expansion!'.format(basis))
        elif callable(basis):
            self.orthogonalBasis = basis
        else:
            raise TypeError('`basis` must be a string or a callable!')
        self.orthogonalBasisParam = basisParam

    def orthogonalSeriesExpansion(self,inputsignal,outputsignal):
        inputsignal = np.asarray(inputsignal)
        outputsignal = np.asarray(outputsignal)
        if callable(self.orthogonalBasisParam):
            param = self.orthogonalBasisParam(inputsignal.size)
        else:
            param = self.orthogonalBasisParam
        numerCoeff = np.zeros(param+1)
        denomCoeff = np.zeros(param+1)
        for i in range(param+1):
            numerCoeff[i] = np.mean(outputsignal*self.orthogonalBasis(i)(inputsignal))
            denomCoeff[i] = np.mean(self.orthogonalBasis(i)(inputsignal))
        if self.shift:
            numerat0 = np.sum([self.orthogonalBasis(i)(0)*numerCoeff[i] for i in range(param+1)])
            denomat0 = np.sum([self.orthogonalBasis(i)(0)*denomCoeff[i] for i in range(param+1)])
            mu0 = numerat0/denomat0
            numerCoeff -= mu0*denomCoeff
        self.orthogonalCoefficients = numerCoeff,denomCoeff
        self.orthogonalNumerator = np.sum([self.orthogonalBasis(i)*numerCoeff[i] for i in range(param+1)])
        self.orthogonalDenominator = np.sum([self.orthogonalBasis(i)*denomCoeff[i] for i in range(param+1)])
        self.orthogonalApproximation = lambda x: self.orthogonalNumerator(x)/self.orthogonalDenominator(x)
