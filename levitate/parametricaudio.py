import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft,ifft,fftfreq
# TODO: Check if the real versions always will work. Might give a performace boost?

class ParametricAudioModel:

	def modulate(self):self._modulate(self)
	def demodulate(self):self._demodulate(self)
	def equalize(self):self._equalize(self)

	def __init__(self,fc=40e3,fs=None, equalize=True,
		demodulation='envelope',modulation='sqrt',depth=0.95,
		padcycles=10, windowcycles=25, lpf=True):
		
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
		self.padding = np.ceil(padcycles*(self.fs/fc)).astype('int')
		self.windowcycles = windowcycles

		if isinstance(lpf,bool):
			if lpf:
				self.lpf = signal.iirfilter(6,2*24e3/self.fs,btype='low')
			else:
				self.lpf = None
		elif isinstance(lpf,tuple):
			self.lpf = lpf
		else:
			raise ValueError('Specify lpf as bool or filter tuple!')
		self.setEqualization(equalize)
		self.setModulation(modulation)
		self.setDemodulation(demodulation)

	def setDemodulation(self,demodulation):
		if demodulation == 'square':
			# Do note that since no 'derivatives' are applied here, any pre-equalization will distort the audio
			def _demodulate(self): 
				self.audiblesound = np.r_[0,np.diff(self.signal**2,2),0]
			self._demodulate = _demodulate
		elif demodulation == 'envelope':
			def _demodulate (self):
				# np.abs required for the exponential modulation: the envelope is complex
				#envfft = fft(np.abs(self.envelope)**2)
				#w=2*np.pi*fftfreq(envfft.size,1/self.fs)
				#demodSignal = np.real(ifft(-w**2 * envfft))
				demodSignal = np.r_[0,np.diff(np.abs(self.envelope)**2,2),0]
				if self.lpf:
					demodSignal = signal.lfilter(self.lpf[0], self.lpf[1], demodSignal)
				self.audiblesound = demodSignal/np.median(np.abs(demodSignal[self.padding:-self.padding]))*self._scale
			self._demodulate = _demodulate
		elif demodulation == 'detection':
			def _demodulate (self):
				env = np.abs(signal.hilbert(self.ultrasound)) 
				self.detectenvelope = env
				envfft = fft(env**2)
				w=2*np.pi*fftfreq(envfft.size,1/self.fs)
				demodSignal = np.real(ifft(-w**2 * envfft))
				if self.lpf:
					demodSignal = signal.lfilter(self.lpf[0], self.lpf[1], demodSignal)
				self.audiblesound = demodSignal/np.median(np.abs(demodSignal[self.padding:-self.padding]))*self._scale
			self._demodulate = _demodulate
		else:
			raise ValueError('Unknown demodulation model: `{}`'.format(demodulate))

	def setModulation(self,modulation):
		if modulation == 'simple':
			def _modulate(self):
				self.envelope = self.depth*self.signal
				self.ultrasound = self.envelope*self.carrier
			self._modulate = _modulate
		elif modulation == 'dsb':
			def _modulate (self):
				self.envelope = 1 + self.depth*self.signal
				self.ultrasound = self.envelope*self.carrier
			self._modulate = _modulate
		elif modulation == 'sqrt':
			def _modulate (self):
				self.envelope = np.sqrt(1 + self.depth*self.signal)
				self.ultrasound = self.envelope*self.carrier
			self._modulate = _modulate
		elif modulation == 'exponential':
			def _modulate(self):
				self.envelope = np.exp(signal.hilbert(np.log(1+self.depth*self.signal))/2)
				self.ultrasound = np.real( self.envelope * signal.hilbert(self.carrier) )
			self._modulate = _modulate
		else:
			raise ValueError('Unknown modulation model: `{}`'.format(modulate))

	def setEqualization(self,equalization):
		if isinstance(equalization,bool):
			if equalization:
				equalization = 'time' # This is be the default method!
			else:
				equalization = 'none'

		if equalization[:2] == 'no':
			def _equalize(self):
				self.signal /= np.max(np.abs(self.signal))
			self._equalize = _equalize
		elif equalization[:4] == 'time' :
			def _equalize(self):
				self.signal = signal.detrend( np.cumsum( signal.detrend( np.cumsum( self.signal )/self.fs ) )/self.fs )
				self.signal /= np.max(np.abs(self.signal))
			self._equalize = _equalize
		elif equalization[:4] == 'freq':
			def _equalize(self):
				fftsignal = fft(self.signal)
				w = 2*np.pi*fftfreq(fftsignal.size,1/self.fs)
				eqfft = np.r_[0,-fftsignal[1:]/(w[1:]**2) ]
				self.signal = np.real(ifft(eqfft))
				self.signal /= np.max(np.abs(self.signal))
			self._equalize = _equalize
		elif equalization[:4] == 'filt':
			def _equalize(self):
				# TODO: How could this be chosen?
				# Choose to a frequency depending on the number of samples?
				cutoff = 10*2/self.signal.size # This gives roughly frequency bin 10
				lpf = signal.iirfilter(2,cutoff,btype='low')
				self.signal = signal.lfilter(lpf[0],lpf[1],self.signal)
				self.signal /= np.max(np.abs(self.signal))
			self._equalize = _equalize
		else:
			raise ValueError('Unknown equalization method: `{}`'.format(equalization))


	def setSignal(self,inputsignal):
		self.signal = inputsignal
		self.eqsignal = inputsignal/np.max(np.abs(inputsignal))
		self._scale = np.median(np.abs(self.signal))
		window = signal.tukey(inputsignal.size, self.smoothcycles*2*self.fs/self.fc/inputsignal.size,sym=True)
		self.window = window
		#self.eqsignal = self.eqsignal*window
		if self.equalize:
			# Note that if no equalization is applied, the signal will be inverted by the differentiation later
			fftsignal = fft(self.eqsignal)
			w = 2*np.pi*fftfreq(fftsignal.size,1/self.fs)
			eqfft = np.r_[0,-fftsignal[1:]/(w[1:]**2) ]
			self.eqsignal = np.real(ifft(eqfft))
			#self.eqsignal = signal.detrend( np.cumsum( signal.detrend( np.cumsum( self.eqsignal )/self.fs ) )/self.fs )
			self.eqsignal = self.eqsignal/np.max(np.abs(self.eqsignal))
		self.eqsignal = np.r_[np.zeros(self.padding), self.eqsignal, np.zeros(self.padding)]
		self.n = self.eqsignal.size
		self.t = (np.arange(self.n)-self.padding)/self.fs
		self.carrier = np.sin(2*np.pi*self.fc*self.t)


	def __call__(self,inputsignal):  
		#self.setSignal(inputsignal)
		self._scale = np.median(np.abs(inputsignal))
		if self.windowcycles>0 :
			self.window = signal.tukey(inputsignal.size, self.windowcycles*2*self.fs/self.fc/inputsignal.size,sym=True)
			self.signal = inputsignal*self.window
		else:
			self.signal =inputsignal.copy()
		self.equalize()
		self.signal = np.r_[np.zeros(self.padding), self.signal, np.zeros(self.padding)]
		self.n = self.signal.size # TODO: Is this really needed?
		self.t = (np.arange(self.n)-self.padding)/self.fs
		self.carrier = np.sin(2*np.pi*self.fc*self.t)
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
	model = ParametricAudioModel(demodulation='detection',fs=fs)
	outputsignal = model(inputsignal)
	plt.plot(model.t,model.signal)
	plt.plot(model.t,model.audiblesound)
	plt.show()

