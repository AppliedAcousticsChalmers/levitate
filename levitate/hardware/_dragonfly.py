import numpy as np
from ..arrays import NormalTransducerArray


class DragonflyArray(NormalTransducerArray):
    """Rectangular array with Ultrahaptics Dragonfly U5 layout.

    This is a 16x16 element array where the order of the transducer elements
    are the same as the iteration order in the Ultrahaptics SDK. Otherwise
    behaves exactly like a `RectangularArray`.
    """

    _str_fmt_spec = '{:%cls(transducer=%transducer, offset=%offset, normal=%normal, rotation=%rotation)}'
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
        positions = np.zeros((3, self.grid_indices.size), float)
        positions[:, self.grid_indices] = np.stack(
            np.meshgrid(np.arange(16) - 7.5, 7.5 - np.arange(16), 0, indexing='xy'),
            axis=0).squeeze() * self.spread
        super().__init__(positions=positions, normals=[0, 0, 1], transducer_size=10e-3, **kwargs)
