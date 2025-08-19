import numpy as np


def filternbpvp(n: int, ratio_wo_panel: float = 0.5) -> float:
    """
    function to output the number of pv panels on a given house based on a random number
    :param n: a random number between 0 and 1
    :param ratio_wo_panel: the ratio of houses without pv panels
    :return: the number of pv panels on the given house
    """
    if n < ratio_wo_panel:  # 50% chance of having 0 pv panels
        nb_pvp = 0
    else:  # n > 0.5
        # 50% chance of having between 6 and 12 panels; 6 + deltaY/deltaX * (12 - 6), line from (0.5, 6) to (1, 12)
        nb_pvp = 6 + (n - ratio_wo_panel) / (1 - ratio_wo_panel) * (12 - 6)

    return nb_pvp

