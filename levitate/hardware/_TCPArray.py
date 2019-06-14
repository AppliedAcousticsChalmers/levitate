import numpy as np
import socket
import subprocess
import os.path


class TCPArray:
    """Communicate with a Ultrahaptics array via TCP.

    Starts a c++ process in the background which connects to a Ultrahaptics array.
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
    def transducer_positions(self):
        """Positions of the transducer elements."""
        num_transducers = self.num_transducers
        self._send('transducer positions')
        raw = self._recv(num_transducers)
        return np.array([np.array(x.decode().strip('()').split(',')).astype(float) for x in raw])

    @property
    def transducer_normals(self):
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
