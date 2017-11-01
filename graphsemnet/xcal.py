from scipy.interpolate import interp1d


def get_xcal(dip_center, dip_width, y_min, y_max):
    """
    Function for differentially reweighting weak vs strong graph edges

    xcal function with symmetrical dip.

    Arguments
    ---------
    dip_center : float [0, 1]
    dip_width : float [0, 1]
    y_min : float [-1, 1]
    y_max : float [-1, 1]

    Returns
    -------
    function [0, 1] -> [y_min, y_max]
    """
    dip_half_width = dip_width / 2
    xp = [
        0,
        dip_center - dip_half_width,
        dip_center,
        dip_center + dip_half_width,
        1
    ]
    fp = [
        0,
        0,
        y_min,
        0,
        y_max
    ]
    return interp1d(xp, fp)


def compute_nmph(min_y=-0.1, inflection_x=0.5, inflection_y=0.05, y_max=0.1):
    """
    Compute Non Monotonic Plasticity Hypothesis function

    Arguments
    ---------
    min_y : float [-1, 1]
        minimum value of the nmph function
    inflection_x : float [0, 1]
        x-coordinate of the point of inflection
    inflection_y : float [-1, 1]
        y-coordinate of the point of inflection
    y_max : float [-1, 1]
        y-max of nmph when x = 1

    Returns
    -------
    nmph : nmph function [0, 1] -> [-1, 1]
    """
    min_x = inflection_x / 2.
    x = [0, min_x, inflection_x, 1.]
    y = [0, min_y, inflection_y, y_max]
    nmph = interp1d(x, y)
    return nmph
