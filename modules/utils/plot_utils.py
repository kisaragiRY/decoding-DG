from numpy.typing import NDArray
from typing import Union

import matplotlib.pyplot as plt

def label_diff(
        i: int,
        j: int, 
        text: str, 
        ind: Union[NDArray, list],
        Y: Union[NDArray, list], 
        errors: Union[NDArray, list], 
        ax: plt.Axes, 
        barh: float
        ) -> None:
    '''Annotate significance bars.

    Parameters:
    ---
    i: int
        index for the first label.
    j: int
        index for the second label.
    text: str
        annotation texts.
    ind: Union[NDArray, list]
        plot's x axis index.
    Y: Union[NDArray, list]
        plot's y axis values.
    errors: Union[NDArray, list]
        the erros for the bar plot.
    ax: plt.Axes
        the axis of the plot.
    barh: float
        the hight of the annotation bar.
    '''
    y = 1.1*max(Y[i]+errors[i], Y[j]+errors[j])

    lx, rx = ind[i], ind[j]
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]

    kwargs = dict(ha='center', va='bottom')
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')
    ax.text(*mid, text, **kwargs)
