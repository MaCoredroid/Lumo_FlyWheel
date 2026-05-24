# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for direct ITRS to/from AltAz and HADec.

This provides a direct approach to ITRS to Observed transformations that stays
within the ITRS frame. This is more accurate for nearby objects (satellites,
buildings, etc.) where the geocentric vs topocentric aberration matters.

The current ITRS->AltAz transform goes through ICRS, which references ITRS
coordinates to the SSB. Since ITRS positions are tied to the Earth, this
causes nearby ITRS positions to be lost in the wake of the Earth's orbit
around the SSB. The direct approach avoids this problem entirely.
"""

import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose

from .itrs import ITRS
from .altaz import AltAz
from .hadec import HADec
from .utils import PIOVER2


def itrs_to_observed_mat(observed_frame):
    """
    Compute rotation matrix from ITRS to the observed frame (AltAz or HADec).

    Parameters
    ----------
    observed_frame : AltAz or HADec frame instance
        The target observed frame with location defined.

    Returns
    -------
    numpy.ndarray
        A (3, 3) or stacked rotation matrix.
    """
    lon, lat, height = observed_frame.location.to_geodetic('WGS84')
    elong = lon.to_value(u.radian)

    if isinstance(observed_frame, AltAz):
        # form ITRS to AltAz matrix
        elat = lat.to_value(u.radian)
        # AltAz frame is left-handed
        minus_x = np.eye(3)
        minus_x[0, 0] = -1.0
        mat = (minus_x
               @ rotation_matrix(PIOVER2 - elat, 'y', unit=u.radian)
               @ rotation_matrix(elong, 'z', unit=u.radian))

    else:
        # form ITRS to HADec matrix
        # HADec frame is left-handed
        minus_y = np.eye(3)
        minus_y[1, 1] = -1.0
        mat = (minus_y
               @ rotation_matrix(elong, 'z', unit=u.radian))
    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference,
                                 ITRS, AltAz, priority=0)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference,
                                 ITRS, HADec, priority=0)
def itrs_to_observed(itrs_coo, observed_frame):
    """
    Transform ITRS coordinates to AltAz or HADec.

    This approach stays entirely within the ITRS frame and does not reference
    ITRS coordinates to the SSB. It treats ITRS positions as time-invariant,
    which is the correct behavior for Earth-tied coordinates.

    The ``obstime`` of the output frame is simply adopted without
    synchronization, since ITRS->ITRS time transforms would incorrectly
    reference ITRS coordinates to the SSB.
    """
    # form the Topocentric ITRS position
    topocentric_itrs_repr = (itrs_coo.cartesian
                             - observed_frame.location.get_itrs().cartesian)
    rep = topocentric_itrs_repr.transform(itrs_to_observed_mat(observed_frame))
    return observed_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference,
                                 AltAz, ITRS, priority=0)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference,
                                 HADec, ITRS, priority=0)
def observed_to_itrs(observed_coo, itrs_frame):
    """
    Transform AltAz or HADec coordinates to ITRS.

    This is the inverse of ``itrs_to_observed``. The transformation stays
    within the ITRS frame.
    """
    # form the Topocentric ITRS position
    topocentric_itrs_repr = observed_coo.cartesian.transform(
        matrix_transpose(itrs_to_observed_mat(observed_coo)))
    # form the Geocentric ITRS position
    rep = topocentric_itrs_repr + observed_coo.location.get_itrs().cartesian
    return itrs_frame.realize_frame(rep)
