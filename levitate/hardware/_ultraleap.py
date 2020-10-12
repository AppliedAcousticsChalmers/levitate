import numpy as np
import socket
import subprocess
import os.path
from ..arrays import NormalTransducerArray


class UltraleapArray:
    """Mixin class for Ultraleap arrays.

    This implements a very simple control of the Ultraleap hardware from pytohn,
    by sending data via TCP to a c++ process running in the background.

    Note
    ----
    This class is not implemented by Ultraleap, and no guarantee is given that
    the implementation works or gives the correct output.

    **Use at your own risk!**

    """
    def connect(self, *args, **kwargs):
        """Connect to a physical array.

        See `UltraleapArray.TCPconnection` for more details.

        """
        self.connection = UltraleapArray.TCPconnection(*args, **kwargs)

    @property
    def connection(self):
        try:
            return self._connection
        except AttributeError:
            self.connect()
            return self._connection

    @connection.setter
    def connection(self, value):
        self._connection = value

    class TCPconnection:
        """Communicate with a Ultraleap array via TCP.

        Starts a c++ process in the background which connects to a Ultraleap array.
        Communicates with the c++ program using TCP messages.
        The only mode implemented for the array is a cyclical transition of stored states.
        A number of states is loaded, either from file or sent via TCP. The states are
        cycled through in a configurable rate or manually. At the end of the list of states
        the cycle starts over from the start.
        This makes it relatively easy to create closed paths by ensuring that the last state
        and the first state is designed to levitate at (almost) the same position.

        The required binary file `array_control` is compiled from the included c++ source files if not present.
        Inspect the makefile in unix systems, or make.bat on windows to see how the files are compiled.
        If the compilation fails a `RuntimeError` is raised. If the binary already exists it will be used.

        Parameters
        ----------
        ip : string, default '127.0.0.1'
            The IP address to use for the local TCP-connection.
        port : int, default 0
            The port to use for the TCP connection.
        use_array : bool, default True
            Set to false to not try to connect to an array, just run the c++
            executable in the background. Mostly for debugging.
        verbose : int, default 0
            Control the verbosity level of the c++ program.
            0 will not print anything, higher values will give more information.
        normalize : bool, default True
            Toggles normalization of the state amplitudes.

        """

        _executable = 'array_control'

        def __init__(self, ip='127.0.0.1', port=0, use_array=True, verbose=0, normalize=True):
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.bind((ip, port))
            self.ip, self.port = self.sock.getsockname()

            args = []
            if verbose:
                args.append('--verbose')
                args.append(str(verbose))
            if not use_array:
                args.append('-noarray')

            self._start_subprocess(*args)
            self.sock.listen()
            self.conn, self._addr = self.sock.accept()
            self.normalize = normalize

        def _start_subprocess(self, *extra_args):
            directory = os.path.dirname(__file__)
            name = os.path.join(directory, self.executable)
            if not os.path.exists(name):
                self._compile()
            args = [name, '--ip', self.ip, str(self.port)]
            args.extend(extra_args)
            self._cpp_process = subprocess.Popen(args=args)

        @staticmethod
        def _compile():
            directory = os.path.dirname(__file__)
            if os.name == 'nt':
                result = subprocess.run(os.path.join(directory, 'make.bat'), cwd=directory)
                if result.returncode != 0:
                    raise RuntimeError('array_control binary non-existent and c++ toolchain cannot compile binary!')
            else:
                result = subprocess.run('make', cwd=directory)
                if result.returncode != 0:
                    raise RuntimeError('array_control binary non-existent and c++ toolchain cannot compile binary!')

        def _send(self, *messages):
            for message in messages:
                if type(message) is not bytes:
                    try:
                        message = bytes(message)
                    except TypeError:
                        message = bytes(message, 'ASCII')
                        if message[-1] is not 0:
                            message += b'\0'  # Properly terminate strings.
                msg_len = np.uint32(len(message))
                self.conn.sendall(msg_len)
                self.conn.sendall(message)

        def _recv(self, count=1):
            if count == 1:
                msg_len = np.squeeze(np.frombuffer(self.conn.recv(4), dtype=np.uint32))
                msg = b''
                while len(msg) < msg_len:
                    msg += self.conn.recv(msg_len)
                return msg
            else:
                return [self._recv() for _ in range(count)]

        def close(self):
            """Close the collection to the array and terminate the c++ process."""
            if self._cpp_process.poll() is None:
                self._send('quit')
                self._cpp_process.wait()
            self.conn.close()
            self.sock.close()

        def __del__(self):
            self.close()

        @property
        def executable(self):  # noqa : D401
            """The name of the binary to use."""
            if os.name == 'nt':
                return self._executable + '.exe'
            else:
                return self._executable

        @executable.setter
        def executable(self, val):
            self._executable = val.rstrip('.exe')

        @property
        def emit(self):
            """Control if the array is emitting or not."""
            self._send('emit')
            return self._recv().decode()

        @emit.setter
        def emit(self, val):
            if val is True or val == 'on':
                self._send('emit on')
            elif val is False or val == 'off':
                self._send('emit off')
            else:
                raise ValueError('Unknown emit state: ' + val)

        @property
        def amplitude(self):
            """Control the overall amplitude scaling of the array."""
            self._send('ampl')
            return np.array(self._recv()).astype(float)

        @amplitude.setter
        def amplitude(self, val):
            if val < 0 or val > 1:
                raise ValueError('Amplitude must not be <0 or >1')
            self._send('amplitude ' + str(val))

        @property
        def rate(self):
            """Control the state transition rate of the array, in Hz."""
            self._send('rate')
            return np.array(self._recv()).astype(float)

        @rate.setter
        def rate(self, val):
            self._send('rate ' + str(val))

        def next(self, count=1):
            """Go to the next state.

            Parameters
            ----------
            count : int
                How many states to move, default 1.

            """
            self._send('next ' + str(count))

        def prev(self, count=1):
            """Go to the previous state.

            Parameters
            ----------
            count : int
                How many states to move, default 1.

            """
            self._send('prev ' + str(count))

        @property
        def num_transducers(self):
            """Number of transducers in the array."""
            self._send('transducer count')
            return np.array(self._recv()).astype(int)

        @property
        def positions(self):
            """Positions of the transducer elements."""
            num_transducers = self.num_transducers
            self._send('transducer positions')
            raw = self._recv(num_transducers)
            return np.array([np.array(x.decode().strip('()').split(',')).astype(float) for x in raw])

        @property
        def normals(self):
            """Normals of the transducer elements."""
            num_transducers = self.num_transducers
            self._send('transducer normals')
            raw = self._recv(num_transducers)
            return np.array([np.array(x.decode().strip('()').split(',')).astype(float) for x in raw])

        @property
        def index(self):
            """Current state index."""
            self._send('index')
            return np.array(self._recv()).astype(int)

        @index.setter
        def index(self, val):
            self._send('index ' + str(val))

        @property
        def states(self):
            """Control the stored states.

            Set this property to send new states to the array.
            Get this property to check what states are stored for the array.
            The states have the shape `(M, N)`, where `M` is the number of states,
            and `N` is the number of transducers in the array.
            """
            num_transducers = self.num_transducers
            self._send('printstates')
            num_states_raw = self._recv()
            num_states = np.array(num_states_raw.decode().strip("Displaying all states.")).astype(int)
            states = []
            for _ in range(num_states):
                header = self._recv()
                state_raw = self._recv(num_transducers)
                states.append([complex(*np.array(trans_raw.decode().rsplit(':')[1].strip(' () ').split(',')).astype(float)) for trans_raw in state_raw])
            return np.array(states).conj()

        @states.setter
        def states(self, states):
            states = np.atleast_2d(states)
            if self.normalize:
                normalization = np.max(np.abs(states))
            else:
                normalization = 1
            num_states = states.size / self.num_transducers
            if not num_states == int(num_states):
                raise ValueError('Cannot send uncomplete states!')
            self._send('states ' + str(num_states))
            for state in states:
                msg = (state / normalization).conj().astype(np.complex64).tobytes()
                self._send(msg)

        def read_file(self, filename):
            """Read a file with states.

            Specify a file with states to read in the c++ process.
            This is not to be confused with `~levitate.hardware.data_from_c++`, which
            reads a file to a numpy.array.

            Parameters
            ----------
            filename : str
                The file to read.

            """
            self._send('file ' + filename)


class DragonflyArray(NormalTransducerArray, UltraleapArray):
    """Rectangular array with Ultraleap Dragonfly U5 layout.

    This is a 16x16 element array where the order of the transducer elements
    are the same as the iteration order in the Ultraleap SDK. Otherwise
    behaves exactly like a `RectangularArray`.
    """

    spread = 10.47e-3
    grid_indices = np.array([
        [95, 94, 93, 92, 111, 110, 109, 108, 159, 158, 157, 156, 175, 174, 173, 172],
        [91, 90, 89, 88, 107, 106, 105, 104, 155, 154, 153, 152, 171, 170, 169, 168],
        [87, 86, 85, 84, 103, 102, 101, 100, 151, 150, 149, 148, 167, 166, 165, 164],
        [83, 82, 81, 80, 99, 98, 97, 96, 147, 146, 145, 144, 163, 162, 161, 160],
        [79, 78, 77, 76, 127, 126, 125, 124, 143, 142, 141, 140, 191, 190, 189, 188],
        [75, 74, 73, 72, 123, 122, 121, 120, 139, 138, 137, 136, 187, 186, 185, 184],
        [71, 70, 69, 68, 119, 118, 117, 116, 135, 134, 133, 132, 183, 182, 181, 180],
        [67, 66, 65, 64, 115, 114, 113, 112, 131, 130, 129, 128, 179, 178, 177, 176],
        [49, 48, 51, 50, 1, 0, 3, 2, 241, 240, 243, 242, 193, 192, 195, 194],
        [53, 52, 55, 54, 5, 4, 7, 6, 245, 244, 247, 246, 197, 196, 199, 198],
        [57, 56, 59, 58, 9, 8, 11, 10, 249, 248, 251, 250, 201, 200, 203, 202],
        [61, 60, 63, 62, 13, 12, 15, 14, 253, 252, 255, 254, 205, 204, 207, 206],
        [33, 32, 35, 34, 17, 16, 19, 18, 225, 224, 227, 226, 209, 208, 211, 210],
        [37, 36, 39, 38, 21, 20, 23, 22, 229, 228, 231, 230, 213, 212, 215, 214],
        [41, 40, 43, 42, 25, 24, 27, 26, 233, 232, 235, 234, 217, 216, 219, 218],
        [45, 44, 47, 46, 29, 28, 31, 30, 237, 236, 239, 238, 221, 220, 223, 222]
    ])

    def __init__(self, **kwargs):
        ny, nx = self.grid_indices.shape
        positions = np.zeros((3, nx * ny), float)
        positions[:, self.grid_indices] = np.stack(
            np.meshgrid(np.arange(16) - 7.5, 7.5 - np.arange(16), 0, indexing='xy'),
            axis=0).reshape((3, ny, nx)) * self.spread
        super().__init__(positions=positions, normals=[0, 0, 1], transducer_size=10e-3, **kwargs)
