# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to/from "observed" systems
(AltAz, HADec) directly from ITRS, staying within the ITRS frame.

This approach treats ITRS positions as time-invariant (Earth-fixed), avoiding
the geocentric vs. topocentric aberration issues that arise when routing
through ICRS/GCRS/CIRS.
"""

import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose

from .altaz import AltAz
from .hadec import HADec
from .itrs import ITRS
from .utils import PIOVER2


def itrs_to_observed_mat(observed_frame):
    """
    Build the rotation matrix from ITRS to the observed frame (AltAz or HADec).

    Parameters
    ----------
    observed_frame : AltAz or HADec frame instance
        The target observed frame (must have a ``location`` attribute).

    Returns
    -------
    mat : `numpy.ndarray`
        A 3x3 rotation matrix that transforms ITRS Cartesian coordinates
        to the observed frame's Cartesian coordinates.
    """
    lon, lat, height = observed_frame.location.to_geodetic('WGS84')
    elong = lon.to_value(u.radian)

    if isinstance(observed_frame, AltAz):
        # Form ITRS to AltAz matrix.
        # AltAz frame is left-handed, so we negate the x-axis.
        elat = lat.to_value(u.radian)
        minus_x = np.eye(3)
        minus_x[0, 0] = -1.0
        mat = (minus_x
               @ rotation_matrix(PIOVER2 - elat, 'y', unit=u.radian)
               @ rotation_matrix(elong, 'z', unit=u.radian))
    else:
        # Form ITRS to HADec matrix.
        # HADec frame is left-handed, so we negate the y-axis.
        minus_y = np.eye(3)
        minus_y[1, 1] = -1.0
        mat = (minus_y
               @ rotation_matrix(elong, 'z', unit=u.radian))
    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, HADec)
def itrs_to_observed(itrs_coo, observed_frame):
    """
    Transform ITRS coordinates to an observed frame (AltAz or HADec).

    This transformation stays entirely within the ITRS frame and treats
    ITRS positions as time-invariant (Earth-fixed). The ``obstime`` of the
    output frame is simply adopted from the target frame, avoiding the
    geocentric vs. topocentric aberration issues that arise when routing
    through ICRS/GCRS/CIRS.
    """
    # form the topocentric ITRS position
    topocentric_itrs_repr = (itrs_coo.cartesian
                             - observed_frame.location.get_itrs().cartesian)
    rep = topocentric_itrs_repr.transform(itrs_to_observed_mat(observed_frame))
    return observed_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs(observed_coo, itrs_frame):
    """
    Transform an observed frame (AltAz or HADec) coordinate to ITRS.

    This transformation stays entirely within the ITRS frame and treats
    ITRS positions as time-invariant (Earth-fixed).
    """
    # form the topocentric ITRS position
    topocentric_itrs_repr = observed_coo.cartesian.transform(
        matrix_transpose(itrs_to_observed_mat(observed_coo)))
    # form the geocentric ITRS position
    rep = topocentric_itrs_repr + observed_coo.location.get_itrs().cartesian
    return itrs_frame.realize_frame(rep)
