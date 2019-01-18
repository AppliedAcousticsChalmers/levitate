import numpy as np
import socket
import subprocess
import os.path


class TCPArray:
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
        if self._cpp_process.poll() is None:
            self._send('quit')
            self._cpp_process.wait()
        self.conn.close()
        self.sock.close()
            

    def __del__(self):
        self.close()

    @property
    def executable(self):
        if os.name == 'nt':
            return self._executable + '.exe'
        else:
            return self._executable

    @executable.setter
    def executable(self, val):
        self._executable = val.rstrip('.exe')
    

    @property
    def emit(self):
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
        self._send('ampl')
        return np.array(self._recv()).astype(float)

    @amplitude.setter
    def amplitude(self, val):
        if val < 0 or val > 1:
            raise ValueError('Amplitude must not be <0 or >1')
        self._send('amplitude ' + str(val))

    @property
    def rate(self):
        self._send('rate')
        return np.array(self._recv()).astype(float)

    @rate.setter
    def rate(self, val):
        self._send('rate ' + str(val))

    def next(self, count=1):
        self._send('next ' + str(count))

    def prev(self, count=1):
        self._send('prev ' + str(count))

    @property
    def num_transducers(self):
        self._send('transducer count')
        return np.array(self._recv()).astype(int)

    @property
    def transducer_positions(self):
        num_transducers = self.num_transducers
        self._send('transducer positions')
        raw = self._recv(num_transducers)
        return np.array([np.array(x.decode().strip('()').split(',')).astype(float) for x in raw])

    @property
    def transducer_normals(self):
        num_transducers = self.num_transducers
        self._send('transducer normals')
        raw = self._recv(num_transducers)
        return np.array([np.array(x.decode().strip('()').split(',')).astype(float) for x in raw])

    @property
    def index(self):
        self._send('index')
        return np.array(self._recv()).astype(int)

    @index.setter
    def index(self, val):
        self._send('index ' + str(val))

    @property
    def states(self):
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
        self._send('file ' + filename)
