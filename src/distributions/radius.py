'''
Radius distribution
'''

import torch
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform, AffineTransform
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all


class Radius(TransformedDistribution):
    r"""
    Creates a distribution for hyperbolic radius parameterized by `loc`,`scale` and `R` where::

        X ~ Normal(loc, scale)
        Y = R * Sigmoid(X) ~ Radius(loc, scale, R)

    Example::

        >>> m = Radius(0., 1., 10.)
        >>> m.rsample()
        tensor(5.4377)

    Args:
        loc (float or Tensor): mean of the base Normal distribution
        scale (float or Tensor): scale of the base Normal distribution
        R (float or Tensor): maximal hyperbolic radius
    """
    arg_constraints = {'loc': constraints.real,
                       'scale': constraints.positive,
                       'R': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, R, validate_args=None):
        self.loc, self.scale, self.R = broadcast_all(loc, scale, R)
        base_dist = Normal(loc, scale)
        transforms = [SigmoidTransform(), AffineTransform(loc=0, scale=R)]
        super(Radius, self).__init__(base_dist, transforms, validate_args=validate_args)
        
    @property
    def mean(self):
        return torch.sigmoid(self.loc)*self.R