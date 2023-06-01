"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
@created by: Shivan
@created on: 9/28/2022 11:37 AM  
"""
import matplotlib.pyplot as plt
import numpy
import numpy as np

x  = np.arange(0, 1.1, .1)

def quad_poly(x, c, b, a=1):
    return c*(1-x)**2+b*(1-x)+a


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

for i in [-.2, -.1, 0, .1, .2]:
    y  = quad_poly(x, 0, i)
    y2 = quad_poly(x, i, 0)
    style = '--' if i == 0 else '-'
    ax1.plot(x, y, 'k', linestyle=style)
    ax2.plot(x, y2, 'k', linestyle=style)

for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
