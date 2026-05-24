# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Contains the transformation functions for getting to/from "observed" systems
directly from ITRS, staying entirely within the ITRS.
"""

import numpy as np

from astropy import units as u
from astropy.coordinates.baseframe import frame_transform_graph
from astropy.coordinates.transformations import FunctionTransformWithFiniteDifference
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose
from astropy.coordinates.representation import CartesianRepresentation

from .altaz import AltAz
from .hadec import HADec
from .itrs import ITRS
from .utils import PIOVER2


def itrs_to_observed_mat(observed_frame):
    """
    Form a rotation matrix from ITRS to an observed frame (AltAz or HADec). """
    lon, lat, height = observed_frame.location.to_geodetic('WGS84')
    elong = lon.to_value(u.radian)

    if isinstance(observed_frame, AltAz):
        elat = lat.to_value(u.radian)
        # AltAz frame is left handed
        minus_x = np.eye(3)
        minus_x[0][0] = -1.0
        mat = (minus_x
               @ rotation_matrix(PIOVER2 - elat, 'y', unit=u.radian)
               @ rotation_matrix(elong, 'z', unit=u.radian))
    else:
        # ITRS to HADec
        # HADec frame is left handed
        minus_y = np.eye(3)
        minus_y[1][1] = -1.0
        mat = (minus_y
               @ rotation_matrix(elong, 'z', unit=u.radian))
    return mat


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, AltAz)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, ITRS, HADec)
def itrs_to_observed(itrs_coo, observed_frame):
    # Treat ITRS coordinates as time invariant. Synchronizing obstimes here
    # would reference ITRS coordinates to the SSB, which defeats the purpose
    # of a topocentric transform.
    # form the Topocentric ITRS position
    topocentric_itrs_repr = (itrs_coo.cartesian
                             - observed_frame.location.get_itrs().cartesian)
    rep = topocentric_itrs_repr.transform(itrs_to_observed_mat(observed_frame))
    return observed_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, AltAz, ITRS)
@frame_transform_graph.transform(FunctionTransformWithFiniteDifference, HADec, ITRS)
def observed_to_itrs(observed_coo, itrs_frame):
    # form the Topocentric ITRS position
    topocentric_itrs_repr = observed_coo.cartesian.transform(
        matrix_transpose(itrs_to_observed_mat(observed_coo)))
    # form the Geocentric ITRS position
    rep = topocentric_itrs_repr + observed_coo.location.get_itrs().cartesian
    return itrs_frame.realize_frame(rep)


# Create loopback transformations
frame_transform_graph._add_merged_transform(AltAz, ITRS, AltAz)
frame_transform_graph._add_merged_transform(HADec, ITRS, HADec)
