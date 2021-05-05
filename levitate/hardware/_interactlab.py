import numpy as np
try:
    import serial
except ModuleNotFoundError as e:
    _serial_import_error = e

from ..arrays import NormalTransducerArray


class AcoustophoreticBoard(NormalTransducerArray):
    spread = 10.47e-3

    _amplitude_values = 65
    _phase_values = 128
    _board_transducers = 256

    grid_indices = np.array([
        [249, 248, 251, 250, 185, 184, 187, 186, 121, 120, 123, 122, 57, 56, 59, 58],
        [253, 252, 254, 255, 189, 188, 190, 191, 125, 124, 126, 127, 61, 60, 62, 63],
        [241, 240, 243, 242, 177, 176, 179, 178, 113, 112, 115, 114, 49, 48, 51, 50],
        [245, 244, 246, 247, 181, 180, 182, 183, 117, 116, 118, 119, 53, 52, 54, 55],
        [233, 232, 235, 234, 169, 168, 171, 170, 105, 104, 107, 106, 41, 40, 43, 42],
        [237, 236, 238, 239, 173, 172, 174, 175, 109, 108, 110, 111, 45, 44, 46, 47],
        [225, 224, 227, 226, 161, 160, 163, 162, 97, 96, 99, 98, 33, 32, 35, 34],
        [229, 228, 230, 231, 165, 164, 166, 167, 101, 100, 102, 103, 37, 36, 38, 39],
        [217, 216, 219, 218, 153, 152, 155, 154, 89, 88, 91, 90, 25, 24, 27, 26],
        [221, 220, 222, 223, 157, 156, 158, 159, 93, 92, 94, 95, 29, 28, 30, 31],
        [209, 208, 211, 210, 145, 144, 147, 146, 81, 80, 83, 82, 17, 16, 19, 18],
        [213, 212, 214, 215, 149, 148, 150, 151, 85, 84, 86, 87, 21, 20, 22, 23],
        [201, 200, 203, 202, 137, 136, 139, 138, 73, 72, 75, 74, 9, 8, 11, 10],
        [205, 204, 206, 207, 141, 140, 142, 143, 77, 76, 78, 79, 13, 12, 14, 15],
        [193, 192, 195, 194, 129, 128, 131, 130, 65, 64, 67, 66, 1, 0, 3, 2],
        [197, 196, 198, 199, 133, 132, 134, 135, 69, 68, 70, 71, 5, 4, 6, 7]
    ])

    _ports = {
        16: '/dev/cu.usbserial-FT4M8GF0',
        17: '/dev/cu.usbserial-FT4YWZWR',
    }
    _phase_calibration = {
        16: np.deg2rad([
            -61, 108, 119, -66, 113, 108, -54, -42, 112, 100, 127, 123, -48, -49, 142, -87,
            -48, 115, -51, 118, 132, 134, 119, 113, -52, -48, -33, 113, -66, 134, -58, 128,
            -64, 123, -56, 127, 124, -52, -51, -57, 107, -60, 121, -33, 128, 105, 122, 129,
            -50, 126, -66, -59, -60, 124, 110, 132, 112, 123, -52, -60, -62, 112, -42, -42,
            -48, 118, 123, 123, 111, -68, 116, -65, 125, 112, -63, -61, -43, -61, 115, 114,
            122, 125, 124, -48, 112, -67, 117, -56, 121, -51, 116, -53, 112, 122, 156, -54,
            125, -51, -62, 131, 111, -68, 125, 127, -69, 119, 121, 116, -57, 126, 127, 138,
            123, 114, -54, 127, -80, -39, 131, -54, -57, -66, -54, 129, 113, -45, -50, -72,
            -63, 120, 122, 125, 100, -43, -47, -66, -84, -69, 128, 116, 117, 111, 117, -66,
            128, 102, 125, -72, 126, -57, 128, -47, -60, 126, -41, 122, -57, -68, 112, -48,
            -45, 118, 125, -40, -54, -61, -48, 118, 131, 124, -65, 106, -50, -56, 123, 120,
            -44, -72, -59, -61, -45, 135, -49, -62, -45, -53, 126, 118, -59, -54, 100, -66,
            -84, 119, -74, -38, -52, 111, 114, 137, 109, -89, 128, 121, 101, 120, -64, 132,
            -67, -61, 116, -38, -42, 125, -55, 141, -59, 121, -58, -50, -63, 115, -45, -60,
            128, -69, -66, 147, -54, 112, 123, 133, 123, -44, 122, 117, 123, 123, 111, 148,
            131, -66, 110, -55, -56, -62, 132, 108, -53, -52, -55, 128, -58, -75, 109, 120,
        ]),
        17: np.deg2rad([
            -152, -153, 37, 29, 55, 49, 51, 30, 43, 51, 37, -149, -136, 20, 41, 47,
            35, -116, 48, -151, -137, 41, -136, 42, 47, 43, 29, 43, 54, 41, -139, -142,
            -139, -138, -142, -149, 52, 55, 49, -121, -141, 59, -140, 43, 39, 54, -125, 52,
            50, 36, -155, 55, 26, 49, -137, -150, 56, -141, -126, 52, -132, 46, -132, -154,
            45, -143, -137, -118, 49, -150, -135, 52, -144, 37, 31, 38, -135, 42, 39, -140,
            29, -118, 45, 25, 56, -137, -150, -144, -138, -127, 50, -133, -153, 43, -139, -147,
            -126, -137, 53, -146, -126, -147, -137, -132, -140, 48, 39, 19, 35, 37, 34, -125,
            41, 43, 45, 55, 54, -138, 34, -134, -138, -135, 64, 55, 45, 36, -168, -124,
            43, 41, 34, 35, -147, -139, 46, -138, 44, -123, -125, 33, 43, -143, 42, 41,
            -149, -148, 50, -151, 37, 28, 45, 33, -145, -140, 41, -139, 42, -131, -157, 36,
            45, -135, -138, -149, -142, 39, 40, 26, -132, -136, -155, 43, 61, 25, -131, 57,
            32, 45, -134, -143, -144, -133, -144, -130, 52, 42, -136, 33, -140, 33, -121, 50,
            -145, 22, -144, -148, 38, 33, -140, -134, -134, -140, -131, 39, -136, -124, 31, -135,
            -141, -128, 33, 56, 54, -136, -170, 44, -138, -149, -150, 56, -150, -137, 53, 40,
            -147, -133, -135, 32, -136, 33, 41, 44, -122, 41, 52, -123, -137, 38, -138, 31,
            25, 35, -148, 42, -138, 40, 38, -133, -136, -123, -130, -139, -151, -137, -124, 35,
        ]),
    }

    _amplitude_calibration = {}

    def __init__(self, id=None, linearize_amplitude=True, compensate_phase=True, normalize=False, use_phase_calibration=True, use_amplitude_calibration=False, **kwargs):
        # Setup for virtual array
        nx = self.grid_indices.shape[1]
        ny = self.grid_indices.shape[0]
        positions = np.zeros((3, nx * ny), float)
        positions[:, self.grid_indices] = np.stack(
            np.meshgrid(np.arange(nx) - (nx - 1) / 2, (ny - 1) / 2 - np.arange(ny), 0, indexing='xy'),
            axis=0).reshape((3, ny, nx)) * self.spread
        kwargs.setdefault('positions', positions)
        kwargs.setdefault('normals', [0, 0, 1])
        kwargs.setdefault('transducer_size', 10e-3)
        super().__init__(**kwargs)

        # Setup for physical arrays
        if isinstance(id, str) or not hasattr(id, '__iter__'):
            id = [id]
        self.id = id
        self.linearize_amplitude = linearize_amplitude
        self.compensate_phase = compensate_phase
        self.normalize = normalize
        self.use_phase_calibration = use_phase_calibration
        self.use_amplitude_calibration = use_amplitude_calibration
        self._connection = {}

    def connect(self):
        try:
            serial
        except NameError:
            raise NameError("Module 'serial' not imported. This is needed to control the hardware from InteractLab. \nTry running `pip install pyserial` in your environment install the package.") from _serial_import_error
        for id in self.id:
            if id not in self._ports:
                raise ConnectionError(f'The interact lab array with id `{self.id}` has no listed port!')
            port = self._ports[id]
            self._connection[id] = serial.Serial(
                port=port, baudrate=250000, bytesize=8, parity='N',
                rtscts=False, dsrdtr=False, stopbits=1
            )
        self.set_state(np.zeros(self.num_transducers, dtype=complex))

    def set_state(self, data):
        # Get the amplitudes and phases
        if np.iscomplexobj(data):
            amps = np.abs(data)
            phases = np.angle(data)
        else:
            phases = data
            amps = np.ones(phases.shape)
        self._state = amps * np.exp(1j * phases)  # TODO: should this be done here or after amplitude limitation? Perhaps even include the quantization in the current state for simulation puurposes? Note that the quantization is on the duty cycle, not the amplitude.
        self._calibrate(amps, phases)
        duty, delay = self._pulse_parameters(amps, phases)
        self._send_message(duty, delay)

    def off(self):
        self._send_message(np.zeros(self.num_transducers), np.zeros(self.num_transducers))

    def _calibrate(self, amplitudes, phases):
        idx = 0
        for id in self.id:
            if id in self._amplitude_calibration and self.use_amplitude_calibration:
                amplitudes[idx:idx + self._board_transducers] /= self._amplitude_calibration[id]
            if id in self._phase_calibration and self.use_phase_calibration:
                phases[idx:idx + self._board_transducers] -= self._phase_calibration[id]
            idx += self._board_transducers

    def _pulse_parameters(self, amplitudes, phases):
        max_amp = np.max(amplitudes)
        if self.normalize and max_amp != 0:
            amplitudes /= np.max(amplitudes)
        else:
            amplitudes = np.clip(amplitudes, 0, 1)

        if self.linearize_amplitude:
            half_duty_cycle = 2 / np.pi * np.arcsin(amplitudes)
        else:
            half_duty_cycle = amplitudes

        delay = phases / (2 * np.pi)
        # The actual phase "zero" point is in the center of the pulse
        if self.compensate_phase:
            delay -= half_duty_cycle * 0.5
        # Wrap delay to positive
        delay -= delay // 1

        return half_duty_cycle, delay

    def _send_message(self, duty_cycles, delays):
        idx = 0
        for id in self.id:
            message = np.zeros(2 * self._board_transducers, dtype='uint8')
            message[:self._board_transducers] = np.clip(np.round(delays[idx:idx + self._board_transducers] * self._phase_values), 0, self._phase_values - 1)
            message[self._board_transducers:] = np.clip(np.round(duty_cycles[idx:idx + self._board_transducers] * self._amplitude_values).astype('uint8'), 0, self._amplitude_values - 1)
            message[0] += 128  # Indicates start of package
            self._connection[id].write(message)
            idx += self._board_transducers

    def __add__(self, other):
        plain_array = super().__add__(other)
        if isinstance(other, AcoustophoreticBoard):
            return AcoustophoreticBoard(
                positions=plain_array.positions, normals=plain_array.normals,
                transducer=plain_array.transducer, id=self.id + other.id
            )
        return plain_array

    def __iadd__(self, other):
        array = super().__iadd__(other)
        if isinstance(other, AcoustophoreticBoard):
            array.id.extend(other.id)
        return array
