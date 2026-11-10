from __future__ import (absolute_import, division, print_function)
import os

try:
    from . import api
    from .api import *
except ImportError:
    # For gfortran+msvc combination, extra shared libraries may exist
    # (stored by numpy.distutils)
    if os.name == "nt":
        # Only the legacy Windows fallback needs setuptools' old resource API.
        # Modern setuptools no longer ships it on the forecast Linux runtime.
        import pkg_resources
        req = pkg_resources.Requirement.parse("wrf-python")
        extra_dll_dir = pkg_resources.resource_filename(req,
                                                        "wrf-python/.libs")
        if os.path.isdir(extra_dll_dir):
            os.environ["PATH"] += os.pathsep + extra_dll_dir

        from . import api
        from .api import *
    else:
        raise


__all__ = []
__all__.extend(api.__all__)
