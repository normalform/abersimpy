"""
make_db.py
"""
import numpy


def make_db(signal: numpy.ndarray,
            dynamic_range: float = 40,
            normalization: numpy.ndarray = None,
            dual: bool = False) -> numpy.ndarray:
    """
    Scales a matrix using a Decibel compression with a chosen dynamic compression.
    :param signal: The signal to be compressed.
    :param dynamic_range: Dynamic range of compression.
        Default is 40 dB. and a positive matrix will be expressed in the range of [-dyn 0]dB.
    :param normalization: External normalization. Default is max(abs(signal(:))).
        If two matrices are to be compared in dB-range, then the largest of
        the two maximum values may be used as a common reference.
    :param dual: If the signal contains both positive and negative entries the values
        will be displayed in the range of +/- dynamic_range dB where -X dB denotes a dB range in
        magnitude in the negative direction. If a purely positive matrix shall be compared to
        a pos/neg matrix, then the dual and a common normalization may be specified to make the
        colors correspond in an image plot. Another interpretation is that a positive matrix may be
        expressed as [0 dynamic_range] dB instead of [-dynamic_range 0] dB which is the default.
    :return: The decibel compressed signal.
    """
    if normalization is None:
        normalization = numpy.max(numpy.abs(signal))

    _index_positive = numpy.where(signal > 0)
    _index_negative = numpy.where(signal < 0)

    if _index_negative[0].size != 0:
        _negative_flag = 1
        _dual = 1
    else:
        _dual = dual

    _signal_db = signal / normalization

    if _dual != 0:
        _signal_db[_index_positive] = 20.0 * numpy.log10(_signal_db[_index_positive])
        _signal_db[_index_negative] = -20.0 * numpy.log10(-_signal_db[_index_negative])
        _index_null = numpy.where(numpy.abs(_signal_db) > dynamic_range)

        _signal_db[_index_positive] = _signal_db[_index_positive] + dynamic_range
        _signal_db[_index_negative] = _signal_db[_index_negative] - dynamic_range

        _signal_db[_index_null] = 0.0
    else:
        _signal_db = 20.0 * numpy.log10(_signal_db)
        _index_null = numpy.where(numpy.abs(_signal_db) > dynamic_range)
        _signal_db[_index_null] = -dynamic_range

    return _signal_db
