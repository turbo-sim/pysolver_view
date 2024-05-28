#!/usr/bin/env python3
import pytest

# Define the list of tests
tests_list = ["test_differentiation.py", "test_optimization.py", "test_nonlinear_system.py"]

# Run pytest when the python script is executed
pytest.main(tests_list + ["-vv"])
# pytest.main([__file__, "-vv"])
# pytest.main([__file__])