import numpy

from diffraction.diffraction import ExactDiffraction, PseudoDifferential


def adjust_equidistant_steps(control,
                             equidistant_steps,
                             num_steps,
                             step_idx):
    _diff_step_idx = numpy.concatenate((numpy.array([0], dtype=int), numpy.diff(step_idx)))
    _num_recalculate = len(numpy.where(_diff_step_idx)[0])
    if control.diffraction_type == ExactDiffraction or \
            control.diffraction_type == PseudoDifferential:
        if control.diffraction_type == ExactDiffraction:
            _diffraction_factor = 1
        else:
            _diffraction_factor = 3
        if _num_recalculate / num_steps < 0.5 / _diffraction_factor:
            _recalculate = True
            if step_idx[0] == 0:
                _equidistant_steps = True
            else:
                _equidistant_steps = equidistant_steps
        else:
            _recalculate = False
            _equidistant_steps = False
    else:
        _recalculate = False
        _equidistant_steps = equidistant_steps

    return _diff_step_idx, _equidistant_steps, _recalculate