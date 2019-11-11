""""
make_banded.py
"""
import numpy


def make_banded(
        vector_a: numpy.ndarray,
        vector_d: numpy.ndarray = None,
        row_major: bool = False) -> numpy.ndarray:
    """
    Returning the diagonals of vector_a specified in vector_d on BLAS banded matrix form.
    :param vector_a:  Matrix to be treated. The matrix is assumed to be of
        standard, column major, Matlab form.
    :param vector_d: Diagonals on vector form (-kl:ku) where 0 denotes main
        diagonal, -kl denotes the kl'th subdiagonal and ku the ku'th superdiagonal
    :param row_major: Used if the matrix is supposed to be in rowmajor storage (C-style).
    :return: The vector_a on banded form.
    """
    if vector_d is None:
        _m, _n = vector_a.shape
        _vector_d = numpy.range(-_m + 1, _n - 1, dtype=int)
    else:
        _vector_d = vector_d

    _num_points_x = numpy.min(vector_a.shape)
    _banded = numpy.zeros((numpy.max(_vector_d.shape), _num_points_x))

    for _index in range(_vector_d):
        _diag = numpy.diag(vector_a, _vector_d[_index])
        _idd = numpy.range(numpy.max(_diag.shape))
        _stidx = numpy.max(numpy.sign(_vector_d[_index]) * (_num_points_x - numpy.max(_diag.shape)),
                           0)
        _banded[_index, _idd + _stidx] = _diag

    if row_major:
        _banded = numpy.transpose(_banded)

    return _banded
