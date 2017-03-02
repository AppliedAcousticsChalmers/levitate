import numpy as np
#from scipy.fftpack import hilbert
import scipy.signal as signal

def demodulate_envelope(envelope):
	'''A model for the non-linear demodulation in air.

	The model is based on direct interpretation of the envelope 
	of the ultrasonic signal. 
	The output is normalised to unity, so no physical interpresation
	of the signal level is possible.
	The output is approximated at the first and last sample using linear
	extrapolation.

	Arguments:
	envelope -- The envelope of the ultrasound signal.
	'''

	dde2 = np.diff(envelope**2,2)
	fullsignal = np.r_[2*dde2[0]-dde2[1],dde2,2*dde2[-1]-dde2[-2]]
	fullsignal = fullsignal/np.max(np.abs(fullsignal))
	return(fullsignal)

def modulate_dsb(signal,fc,fs,depth=1):
	n=signal.size
	t=np.r_[0:n]/fs
	carrier=np.sin(2*np.pi*fc*t)
	env=1+depth*signal
	return(evn*carrier)

def modulate_sqrt(signal,fc,fs,depth=1):
	n=signal.size
	t=np.r_[0:n]/fs
	carrier=np.sin(2*np.pi*fc*t)
	env=np.sqrt(1+depth*signal)
	return(env*carrier)

class ParametricAudioModel:

	def modulate(self):self._modulate(self)
	def demodulate(self):self._demodulate(self)

	def __init__(self,fc=40e3,fs=None,
		demodulation='envelope',modulation='sqrt',depth=0.95,
		padcycles=10, smoothcycles=25, lpf=True):
		
		self.fc=fc
		if fs==None:
			self.fs = 2*2.4*fc # Gives 192kHz from 40kHz
		else:
			self.fs=fs
			if fs<2*fc:
				raise ValueError('Sampling frequency lower than the nyquist frequency!')
			elif fs<2.56*fc:
				raise RuntimeWarning('Sampling frequency close to the nyquist frequency!')

		self.depth=depth
		self.padding = np.ceil(padcycles*(fs/fc)).astype('int')
		self.smoothcycles = smoothcycles

		if isinstance(lpf,bool):
			if lpf:
				self.lpf = signal.iirfilter(6,24e3/(self.fs/2),btype='low')
			else:
				self.lpf = None
		elif isinstance(lpf,tuple):
			self.lpf = lpf
		else:
			raise ValueError('Specify lpf as bool or filter tuple!')


		if demodulation == 'envelope':
			def _demodulate (self):
				ddenv2 = np.diff(self.envelope**2,2)
				fullsignal = np.r_[0, ddenv2, 0]
				#fullsignal = fullsignal[self.padding:-(self.padding+1)]
				#fullsignal = fullsignal/np.max(np.abs(fullsignal[self.padding:-(self.padding+1)]))
				fullsignal = fullsignal/np.median(np.abs(fullsignal[self.padding:-self.padding]))/np.sqrt(2)
				fullsignal = fullsignal/np.median(np.abs(fullsignal[self.padding:-self.padding]))*self._scale
				self.audiblesound = fullsignal
			self._demodulate = _demodulate
		elif demodulation == 'detection':
			def _demodulate (self):
				env = np.abs(signal.hilbert(self.ultrasound)) 
				self.detectenvelope = env
				ddenv2 = np.diff( env**2 ,2)
				fullsignal = np.r_[0, ddenv2, 0] 
				if self.lpf:
					fullsignal = signal.lfilter(self.lpf[0], self.lpf[1], fullsignal)
				#fullsignal = fullsignal[self.padding:-(self.padding+1)]
				self.audiblesound = fullsignal/np.median(np.abs(fullsignal[self.padding:-self.padding]))*self._scale
			self._demodulate = _demodulate
		else:
			raise ValueError('Unknown demodulation model:', demodulate)

		if modulation == 'dsb':
			def _modulate (self):
				self.envelope = 1 + self.depth*self.eqsignal
				self.ultrasound = self.envelope*self.carrier
			self._modulate = _modulate
		elif modulation == 'sqrt':
			def _modulate (self):
				self.envelope = np.sqrt(1 + self.depth*self.eqsignal)
				self.ultrasound = self.envelope*self.carrier
			self._modulate = _modulate
		else:
			raise ValueError('Unknown modulation model:',modulate)

	def set_signal(self,inputsignal):
		self.signal = inputsignal
		self.eqsignal = inputsignal/np.max(np.abs(inputsignal))
		self.eqsignal = signal.detrend( np.cumsum( signal.detrend( np.cumsum( self.eqsignal )/self.fs ) )/self.fs )
		self.eqsignal = self.eqsignal/np.max(np.abs(self.eqsignal))
		self._scale = np.median(np.abs(self.signal))
		window = signal.tukey(inputsignal.size, self.smoothcycles*2*self.fs/self.fc/inputsignal.size,sym=True)
		self.window = window
		self.eqsignal = self.eqsignal*window
		self.eqsignal = np.r_[np.zeros(self.padding), self.eqsignal, np.zeros(self.padding)]
		self.n = self.eqsignal.size
		self.t = (np.r_[0:self.n]-self.padding)/self.fs
		self.carrier = np.sin(2*np.pi*self.fc*self.t)


	def simulate_path(self,inputsignal):
		self.set_signal(inputsignal)
		self.modulate()
		self.demodulate()
		return(self.audiblesound[self.padding:-self.padding])

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	n=2**12
	f0 = 1e3
	fs=402e3
	fs=204.8e3
	inputsignal = np.sin(2*np.pi*f0*np.r_[0:n]/fs)
	model = ParametricAudioModel(demodulation='detection',fs=400e3)
	outputsignal = model.simulate_path(inputsignal)
	plt.plot(model.t,model.signal)
	plt.plot(model.t,model.audiblesound)
	plt.show()

