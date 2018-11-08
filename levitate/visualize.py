import numpy as np
import matplotlib.pyplot as plt

def transducers(array, transducers='all', projection='xy', transducer_size=10e-3,
                amplitudes=(True, 0, 1), phases=(True, -1, 1), phases_alpha=False,
                amplitudes_colormap='viridis', phases_colormap='hsv',
                labels=True, colorbar=True):
        ''' Visualizes the transducer grid and the amplitudes and phases

        Parameters
        ----------
        transducers : string or iterable
            Controlls which transducers should be visualized. Use an iterable
                for explicit controll. The strings 'all' and 'first-half', and
                'last-half' can also be used.
        projection : string
            Specifies how the transducer locations will be projected. One of:
                'xy', 'xz', 'yz', 'yx', 'zx', 'zy', '3d'
        amplitudes : bool, callable, or tuple
            Toggles if the amplitudes should be displayed.
                Pass a callable which will be applied to the amplitudes.
                Pass a tuple `(amplitudes, v_min, v_max)` with `amplitudes` as
                described, `v_min`, `v_max` sets the plot limits.
        phases : bool, callable, or tuple
            Toggles if the phases should be displayed.
                Pass a callable which will be applied to the phases.
                Defaults to normalize the phases by pi.
                Pass a tuple `(phases, v_min, v_max)` with `phases` as
                described, `v_min`, `v_max` sets the plot limits.
        phases_alpha : bool, callable, or tuple
            Toggles if the phases shuld use alpha values from the amplitudes.
                Pass a callable which will be applied to the amplitudes
                to calculate the alpha value.
                Default False, pass True to use the amplitude as alpha.
                Pass a tuple `(phases_alpha, v_min, v_max)` with `phases_alpha`
                as described, `v_min`, `v_max` sets the alpha limits.
        transducer_size : float
            The diameter of the transducers to visualize. Defaults to 10mm.
        amplitudes_colormap: string
            Which matplotlib colormap to use to the amplitude plot. Default 'viridis'.
        phases_colormap: string
            Which matplotlib colormap to use to the phase plot. Default 'hsv'.
        labels: bool
            Toggles if the transducers should be labled in the figure. Default True.
        colorbar: bool
            Toggles if a colorbar should be drawn. Default True.

        '''
        
        if transducers == 'all':
            transducers = range(array.num_transducers)
        if transducers == 'first_half':
            transducers = range(int(array.num_transducers / 2))
        if transducers == 'last_half':
            transducers = range(int(array.num_transducers / 2), array.num_transducers)

        # Prepare polygon shape creation
        radius = transducer_size / 2
        num_points = 50  # This is the points per half-circle
        theta = np.concatenate([np.linspace(0, np.pi, num_points), np.linspace(np.pi, 2 * np.pi, num_points)])
        cos, sin = np.cos(theta), np.sin(theta)
        if projection == '3d':
            axes = [0, 1, 2]
            def edge(t_idx):
                pos = array.transducer_positions[t_idx]
                norm = array.transducer_normals[t_idx]
                v1 = np.array([1., 1., 1.])
                v1[2] = -(v1[0] * norm[0] + v1[1] * norm[1]) / norm[2]
                v1 /= np.sqrt(np.sum(v1**2))
                v2 = np.cross(v1, norm)

                v1.shape = (-1, 1)
                v2.shape = (-1, 1)
                return (radius * (cos * v1 + sin * v2) + pos[:, np.newaxis]).T
        else:
            axes = [0 if ax == 'x' else 1 if ax == 'y' else 2 if ax == 'z' else 3 for ax in projection]

            def edge(t_idx):
                pos = array.transducer_positions[t_idx][axes]
                return pos + radius * np.stack([cos, sin], 1)
        # Calculate the actual polygons
        verts = [edge(t_idx) for t_idx in transducers]

        # Set the max and min of the scales
        try:
            phases, phase_min, phase_max = phases
        except TypeError:
            phase_min, phase_max = -1, 1
        try:
            amplitudes, amplitude_min, amplitude_max = amplitudes
        except TypeError:
            amplitude_min, amplitude_max = 0, 1
        try:
            phases_alpha, phase_alpha_min, phase_alpha_max = phases_alpha
        except TypeError:
            phase_alpha_min, phase_alpha_max = None, None
        phase_norm = plt.Normalize(phase_min, phase_max)
        amplitude_norm = plt.Normalize(amplitude_min, amplitude_max)
        phase_alpha_norm = plt.Normalize(phase_alpha_min, phase_alpha_max, clip=True)

        # Define default plotting scale
        if phases is True:
            def phases(phase): return phase / np.pi
        if amplitudes is True:
            def amplitudes(amplitude): return amplitude
        if phases_alpha is True:
            def phases_alpha(amplitude): return amplitude

        # Create the colors of the polygons
        two_plots = False
        if not amplitudes and not phases:
            colors = ['blue'] * len(verts)
            colorbar = False
        elif not amplitudes:
            colors = plt.get_cmap(phases_colormap)(phase_norm(phases(array.phases[transducers])))
            norm = phase_norm
            colormap = phases_colormap
            if phases_alpha:
                colors[:, 3] = phase_alpha_norm(phases_alpha(array.amplitudes[transducers]))
        elif not phases:
            colors = plt.get_cmap(amplitudes_colormap)(amplitude_norm(amplitudes(array.amplitudes[transducers])))
            norm = amplitude_norm
            colormap = amplitudes_colormap
        else:
            two_plots = True
            colors_phase = plt.get_cmap(phases_colormap)(phase_norm(phases(array.phases[transducers])))
            colors_amplitude = plt.get_cmap(amplitudes_colormap)(amplitude_norm(amplitudes(array.amplitudes[transducers])))
            if phases_alpha:
                colors_phase[:, 3] = phase_alpha_norm(phases_alpha(array.amplitudes[transducers]))

        if projection == '3d':
            # 3D plots
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            if two_plots:
                ax_amplitude = plt.subplot(1, 2, 1, projection='3d')
                ax_phase = plt.subplot(1, 2, 2, projection='3d')
                ax_amplitude.add_collection3d(Poly3DCollection(verts, facecolors=colors_amplitude))
                ax_phase.add_collection3d(Poly3DCollection(verts, facecolors=colors_phase))
                ax = [ax_amplitude, ax_phase]
            else:
                ax = plt.gca(projection='3d')
                ax.add_collection3d(Poly3DCollection(verts, facecolors=colors))
                ax = [ax]
            xlim = np.min(array.transducer_positions[transducers, 0]) - radius, np.max(array.transducer_positions[transducers, 0]) + radius
            ylim = np.min(array.transducer_positions[transducers, 1]) - radius, np.max(array.transducer_positions[transducers, 1]) + radius
            zlim = np.min(array.transducer_positions[transducers, 2]) - radius, np.max(array.transducer_positions[transducers, 2]) + radius
            for a in ax:
                a.set_xlim3d(xlim)
                a.set_ylim3d(ylim)
                a.set_zlim3d(zlim)
        else:
            # 2d plots, will not actually project transducer positons with cosiderations of the orientation.
            from matplotlib.collections import PolyCollection
            ax0_lim = np.min(array.transducer_positions[transducers, axes[0]]) - radius, np.max(array.transducer_positions[transducers, axes[0]]) + radius
            ax1_lim = np.min(array.transducer_positions[transducers, axes[1]]) - radius, np.max(array.transducer_positions[transducers, axes[1]]) + radius
            if two_plots:
                ax_amplitude = plt.subplot(1, 2, 1)
                ax_phase = plt.subplot(1, 2, 2)
                ax_amplitude.add_collection(PolyCollection(verts, facecolors=colors_amplitude))
                ax_phase.add_collection(PolyCollection(verts, facecolors=colors_phase))
                ax = [ax_amplitude, ax_phase]
            else:
                ax = plt.gca()
                ax.add_collection(PolyCollection(verts, facecolors=colors))
                ax = [ax]
            for a in ax:
                a.set_xlim(ax0_lim)
                a.set_ylim(ax1_lim)
                a.axis('scaled')
                a.grid(False)

            # Create colorbars, does not work for 3d plots
            if colorbar:
                if two_plots:
                    sm_amplitude = plt.cm.ScalarMappable(norm=amplitude_norm, cmap=amplitudes_colormap)
                    sm_amplitude.set_array([])
                    plt.colorbar(sm_amplitude, ax=ax_amplitude, orientation='horizontal')
                    sm_phase = plt.cm.ScalarMappable(norm=phase_norm, cmap=phases_colormap)
                    sm_phase.set_array([])
                    plt.colorbar(sm_phase, ax=ax_phase, orientation='horizontal')
                else:
                    sm = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
                    sm.set_array([])
                    plt.colorbar(sm, ax=ax[0], orientation='horizontal')

            # label the transducers, does not work for 3d plots
            if labels:
                for a in ax:
                    for t_idx in transducers:
                        pos = array.transducer_positions[t_idx][axes]
                        a.text(*pos, str(t_idx))
        return ax
