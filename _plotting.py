import numpy as np
from matplotlib.transforms import blended_transform_factory


def bin_in_ell(l, cl, lbin):
    return np.array([np.mean(cl[(ll <= l) & (l <= lu)]) for ll, lu in zip(lbin, lbin[1:])])


def getcl(cls, i, j, lmax=None):
    if j > i:
        i, j = j, i
    cl = cls[i*(i+1)//2+i-j]
    if lmax is not None:
        cl = cl[:lmax+1]
    return cl


def split_bins(a):
    s = []
    i, j = 0, 0
    while i < len(a):
        j += 1
        s.append(a[i:i+j])
        i += j
    return s


def multi_row_label(fig, ax):
    yl = ax.yaxis.get_label()
    ax_ = fig.add_subplot(111)
    ax_.axis('off')
    tf = blended_transform_factory(yl.get_transform(), ax_.get_yaxis_transform())
    yl.set_transform(tf)
    yl.set_in_layout(False)


def multi_col_label(fig, ax):
    xl = ax.xaxis.get_label()
    ax_ = fig.add_subplot(111)
    ax_.axis('off')
    tf = blended_transform_factory(ax_.get_xaxis_transform(), xl.get_transform())
    xl.set_transform(tf)
    xl.set_in_layout(False)


def dont_draw_zero_tick(tick):
    draw = tick.draw

    def wrap(*args, **kwargs):
        if tick.get_loc() == 0.:
            tick.set_label('')
        draw(*args, **kwargs)

    return wrap


def symlog_no_zero(axes):
    for ax in np.reshape(axes, -1):
        for tick in ax.yaxis.get_major_ticks():
            tick.draw = dont_draw_zero_tick(tick)
