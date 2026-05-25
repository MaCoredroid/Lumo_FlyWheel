# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to/from ITRS and observed
systems (AltAz, HADec). This is a direct approach that stays entirely within
the ITRS, avoiding the geocentric vs. topocentric aberration issues of the
ICRS-based path.
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
    observed_frame : AltAz or HADec frame
        The target observed frame (used only for its ``location`` attribute).

    Returns
    -------
    mat : ndarray
        3x3 rotation matrix from ITRS to the observed frame.
    """
    lon, lat, height = observed_frame.location.to_geodetic('WGS84')
    elong = lon.to_value(u.radian)

    if isinstance(observed_frame, AltAz):
        # AltAz frame is left-handed: az increases eastward from north,
        # alt increases upward from the horizon.
        elat = lat.to_value(u.radian)
        # First rotate around Z by longitude, then around Y by (90-lat)
        # to tilt from ITRS to local horizontal.
        # The minus_x makes the frame left-handed (az increases eastward).
        minus_x = np.eye(3)
        minus_x[0, 0] = -1.0
        mat = (minus_x
               @ rotation_matrix(PIOVER2 - elat, 'y', unit=u.radian)
               @ rotation_matrix(elong, 'z', unit=u.radian))
    else:
        # HADec frame is left-handed: ha increases westward from meridian,
        # dec increases northward from equator.
        # Only rotate around Z by longitude; the minus_y makes the frame
        # left-handed (ha increases westward).
        minus_y = np.eye(3)
        minus_y[1, 1] = -1.0
        mat = (minus_y
               @ rotation_matrix(elong, 'z', unit=u.radian))
    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, HADec)
def itrs_to_observed(itrs_coo, observed_frame):
    """
    Transform ITRS coordinates directly to an observed frame (AltAz or HADec).

    This approach stays entirely within the ITRS and merely converts between
    ITRS, AltAz, and HADec coordinates. The ITRS position is treated as time
    invariant, avoiding the geocentric vs. topocentric aberration issues.

    The ``obstime`` of the output frame is simply adopted (not synchronized
    with the input), since ITRS coordinates are tied to the rotating Earth
    and should not be referenced to the SSB.
    """
    # Form the topocentric ITRS position
    topocentric_itrs_repr = (itrs_coo.cartesian
                             - observed_frame.location.get_itrs().cartesian)

    # Rotate from topocentric ITRS to the observed frame
    rep = topocentric_itrs_repr.transform(itrs_to_observed_mat(observed_frame))
    return observed_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs(observed_coo, itrs_frame):
    """
    Transform observed frame (AltAz or HADec) coordinates directly to ITRS.

    This is the inverse of ``itrs_to_observed``. It takes the topocentric
    observed coordinates, rotates back to topocentric ITRS, and then adds
    the observer's geocentric ITRS position.
    """
    # Rotate from the observed frame back to topocentric ITRS
    topocentric_itrs_repr = observed_coo.cartesian.transform(
        matrix_transpose(itrs_to_observed_mat(observed_coo)))

    # Add the observer's geocentric ITRS position
    rep = topocentric_itrs_repr + observed_coo.location.get_itrs().cartesian
    return itrs_frame.realize_frame(rep)


# Create loopback transformations
frame_transform_graph._add_merged_transform(AltAz, ITRS, AltAz)
frame_transform_graph._add_merged_transform(HADec, ITRS, HADec)
