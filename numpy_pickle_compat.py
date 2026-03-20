import logging


def _normalize_bit_generator_name(value):
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("<class '") and raw.endswith("'>"):
            raw = raw[len("<class '"):-2]
        if "." in raw:
            raw = raw.rsplit(".", 1)[-1]
        return raw
    name = getattr(value, "__name__", None)
    if name:
        return name
    return value


class _CompatBitGenerator:
    def __init__(self, bit_generator_name="MT19937"):
        import numpy as np
        normalized = _normalize_bit_generator_name(bit_generator_name) or "MT19937"
        ctor = getattr(np.random, normalized, None) or np.random.MT19937
        self._bitgen = ctor()
        self._raw_state = None

    def __getattr__(self, name):
        return getattr(self._bitgen, name)

    def __setstate__(self, state):
        self._raw_state = state
        try:
            self._bitgen.state = state
        except Exception:
            pass


class _CompatRandomState:
    def __init__(self, bit_generator_name="MT19937"):
        import numpy as np
        self._rng = np.random.RandomState()
        self._bit_generator_name = _normalize_bit_generator_name(bit_generator_name) or "MT19937"
        self._raw_state = None

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def __setstate__(self, state):
        self._raw_state = state
        try:
            self._rng.set_state(state)
        except Exception:
            pass


class _CompatGenerator:
    def __init__(self, bit_generator_name="MT19937"):
        import numpy as np
        normalized = _normalize_bit_generator_name(bit_generator_name) or "MT19937"
        ctor = getattr(np.random, normalized, None) or np.random.MT19937
        self._generator = np.random.Generator(ctor())
        self._raw_state = None

    def __getattr__(self, name):
        return getattr(self._generator, name)

    def __setstate__(self, state):
        self._raw_state = state
        try:
            if hasattr(self._generator, "bit_generator"):
                self._generator.bit_generator.state = state
        except Exception:
            pass


def _compat_bit_generator_ctor(bit_generator_name="MT19937"):
    normalized = _normalize_bit_generator_name(bit_generator_name)
    return _CompatBitGenerator(normalized)


def _compat_randomstate_ctor(bit_generator_name="MT19937",
                             bit_generator_ctor=None):
    return _CompatRandomState(bit_generator_name)


def _compat_generator_ctor(bit_generator_name="MT19937",
                           bit_generator_ctor=None):
    return _CompatGenerator(bit_generator_name)


def patch_numpy_pickle_compat() -> bool:
    try:
        import numpy.random._pickle as np_pickle
    except Exception:
        return False

    if getattr(np_pickle, "_wstrader_bitgen_patch", False):
        return True

    np_pickle.__bit_generator_ctor = _compat_bit_generator_ctor
    np_pickle.__randomstate_ctor = _compat_randomstate_ctor
    np_pickle.__generator_ctor = _compat_generator_ctor
    np_pickle._wstrader_bitgen_patch = True
    logging.getLogger(__name__).info("NumPy pickle compatibility patch enabled")
    return True


patch_numpy_pickle_compat()