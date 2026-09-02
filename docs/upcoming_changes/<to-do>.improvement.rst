Add platform-wise ``signext`` / ``zeroext`` attributes
------------------------------------------------------

Integer parameters and returns on C-facing functions now get these
attributes per platform. Applies to Python C API
declarations, ``@cfunc`` wrappers, and other foreign functions.
Numba's internal calling convention is unchanged.
