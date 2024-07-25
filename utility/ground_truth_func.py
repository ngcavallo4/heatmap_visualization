import numpy as np

def ground_truth_func(x,y):
    r"""Ground truth function. Can be changed.

        Parameters
        ----------
        x: :class:`np.ndarray`
            X position data.
        y: :class:`np.ndarray`
            y position data.

        Returns
        -------
        g_t_f: :class:`callable`
            Function that performs element-wise operation on ndarray 
            inputs.

    """

    g_t_f = np.sin(x)

    return g_t_f