# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for direct ITRS to observed (AltAz,
HADec) coordinate systems.

These transforms stay entirely within the ITRS and merely convert between
ITRS, AltAz, and HADec coordinates.  This avoids the problematic geocentric
versus topocentric aberration issues that arise when going through the SSB.

ITRS positions are treated as time invariant in these transforms, since
doing an ITRS->ITRS transform for differing obstimes between input and
output frame makes no sense.  The obstime of the output frame is simply
adopted.
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
    Create a rotation matrix from ITRS to the observed frame (AltAz or HADec).

    Parameters
    ----------
    observed_frame : AltAz or HADec frame
        The target observed frame, which must have a location set.

    Returns
    -------
    mat : `~numpy.ndarray`
        The 3x3 rotation matrix from ITRS to the observed frame.
    """
    lon, lat, height = observed_frame.location.to_geodetic('WGS84')
    elong = lon.to_value(u.radian)

    if isinstance(observed_frame, AltAz):
        # form ITRS to AltAz matrix
        # AltAz frame is left handed
        elat = lat.to_value(u.radian)
        minus_x = np.eye(3)
        minus_x[0][0] = -1.0
        mat = (minus_x
               @ rotation_matrix(PIOVER2 - elat, 'y', unit=u.radian)
               @ rotation_matrix(elong, 'z', unit=u.radian))
    else:
        # form ITRS to HADec matrix
        # HADec frame is left handed
        minus_y = np.eye(3)
        minus_y[1][1] = -1.0
        mat = (minus_y
               @ rotation_matrix(elong, 'z', unit=u.radian))
    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, HADec)
def itrs_to_observed(itrs_coo, observed_frame):
    """
    Transform ITRS coordinates to an observed frame (AltAz or HADec).

    This transform stays entirely within the ITRS, treating ITRS coordinates
    as time invariant.  The obstime of the output frame is simply adopted
    from the observed_frame, even if it is None.

    Parameters
    ----------
    itrs_coo : ITRS coordinate
        The input ITRS coordinates.
    observed_frame : AltAz or HADec frame
        The target observed frame.

    Returns
    -------
    observed_coo : AltAz or HADec coordinate
        The coordinates in the observed frame.
    """
    # form the Topocentric ITRS position
    topocentric_itrs_repr = (itrs_coo.cartesian
                             - observed_frame.location.get_itrs().cartesian)
    rep = topocentric_itrs_repr.transform(itrs_to_observed_mat(observed_frame))
    return observed_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs(observed_coo, itrs_frame):
    """ Transform observed coordinates (AltAz or HADec) to ITRS.

    Parameters
    ----------
    observed_coo : AltAz or HADec coordinate
        The input observed coordinates.
    itrs_frame : ITRS frame
        The target ITRS frame.

    Returns
    -------
    itrs_coo : ITRS coordinate
        The coordinates in the ITRS frame.
    """
    # form the Topocentric ITRS position
    topocentric_itrs_repr = observed_coo.cartesian.transform(
        matrix_transpose(itrs_to_observed_mat(observed_coo)))
    # form the Geocentric ITRS position
    rep = topocentric_itrs_repr + observed_coo.location.get_itrs().cartesian
    return itrs_frame.realize_frame(rep)


# Create loopback transformations
frame_transform_graph._add_merged_transform(AltAz, ITRS, AltAz)
frame_transform_graph._add_merged_transform(HADec, ITRS, HADec)
