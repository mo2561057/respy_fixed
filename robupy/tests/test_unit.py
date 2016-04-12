""" This modules contains some additional tests that are only used in long-run
development tests.
"""

# standard library
import numpy as np
import pytest
import scipy
import statsmodels.api as sm
from scipy.optimize import approx_fprime
from scipy.optimize import rosen
from scipy.optimize import rosen_der
from scipy.optimize.slsqp import _minimize_slsqp
from scipy.stats import norm

# testing library
from codes.auxiliary import write_interpolation_grid

from robupy.python.solve.solve_auxiliary import get_payoffs
from robupy.tests.codes.random_init import generate_init

from robupy.python.solve.solve_ambiguity import get_payoffs_ambiguity
from robupy.python.solve.solve_ambiguity import criterion_ambiguity
from robupy.python.solve.solve_ambiguity import divergence
from robupy.python.solve.solve_emax import simulate_emax

from robupy.python.estimate.estimate_auxiliary import get_optim_paras
from robupy.python.estimate.estimate_auxiliary import dist_optim_paras

from robupy.python.shared.shared_auxiliary import dist_class_attributes
from robupy.python.shared.shared_auxiliary import dist_model_paras
from robupy.python.shared.shared_auxiliary import create_draws

import robupy.fortran.f2py_testing as fort_test
import robupy.fortran.f2py_library as fort_lib
import robupy.fortran.f2py_debug as fort_debug

from robupy.python.solve.solve_auxiliary import pyth_create_state_space
from robupy.fortran.f2py_library import f2py_create_state_space

from robupy.python.solve.solve_auxiliary import pyth_calculate_payoffs_systematic
from robupy.fortran.f2py_library import f2py_calculate_payoffs_systematic

from robupy.python.solve.solve_auxiliary import pyth_backward_induction
from robupy.fortran.f2py_library import f2py_backward_induction

from robupy import simulate
from robupy import solve
from robupy import read


''' Main
'''


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
class TestClass(object):
    """ This class groups together some tests.
    """
    def test_1(self):
        """ This function compares the results from the payoff functions across
        implementations.
        """
        for _ in range(5):

            # Generate random initialization file
            generate_init()

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')
            robupy_obj = solve(robupy_obj)

            # Extract class attributes
            periods_payoffs_systematic, states_number_period, \
                mapping_state_idx, is_deterministic, is_ambiguous, \
                periods_emax, model_paras, num_periods, states_all, \
                num_draws_emax, edu_start, is_debug, edu_max, delta,\
                level = dist_class_attributes(robupy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'is_deterministic', 'is_ambiguous',
                    'periods_emax', 'model_paras', 'num_periods', 'states_all',
                    'num_draws_emax', 'edu_start', 'is_debug', 'edu_max',
                    'delta', 'level')

            # Extract auxiliary objects
            _, _, _, _, shocks_cov, shocks_cholesky = \
                dist_model_paras(model_paras, is_debug)

            # Iterate over a couple of admissible points
            for _ in range(10):

                # Select random points
                period = np.random.choice(range(num_periods))
                k = np.random.choice(range(states_number_period[period]))

                # Finalize extraction of ingredients
                payoffs_systematic = periods_payoffs_systematic[period, k, :]
                draws_emax = np.random.sample((num_draws_emax, 4))

                # Extract payoffs using PYTHON and FORTRAN codes.
                py = get_payoffs(num_draws_emax, draws_emax, period, k,
                    payoffs_systematic, edu_max, edu_start,
                    mapping_state_idx, states_all, num_periods,
                    periods_emax, delta, is_debug, shocks_cov, level,
                    is_ambiguous, is_deterministic,
                    shocks_cholesky)

                f90 = fort_debug.wrapper_get_payoffs(num_draws_emax,
                    draws_emax, period, k, payoffs_systematic,
                    edu_max, edu_start, mapping_state_idx, states_all,
                    num_periods, periods_emax, delta, is_debug, shocks_cov,
                    level, is_ambiguous, is_deterministic,
                    shocks_cholesky)

                # Compare returned array on expected future values, ex post
                # payoffs, and future payoffs.
                np.testing.assert_array_almost_equal(py, f90)

    def test_2(self):
        """ This test compares the functions calculating the payoffs under
        ambiguity.
        """
        # Iterate over random test cases
        for _ in range(5):

            # Generate constraint periods
            constraints = dict()
            constraints['level'] = 0.0
            constraints['version'] = 'PYTHON'

            # Generate random initialization file
            generate_init(constraints)

            # Perform toolbox actions
            robupy_obj = read('test.robupy.ini')
            robupy_obj = solve(robupy_obj)

            # Extract class attributes
            periods_payoffs_systematic, states_number_period, \
                mapping_state_idx, is_deterministic, periods_emax, \
                num_periods, model_paras, states_all, num_draws_emax, \
                edu_start, edu_max, delta, is_debug = \
                    dist_class_attributes(robupy_obj,
                        'periods_payoffs_systematic', 'states_number_period',
                        'mapping_state_idx', 'is_deterministic', 'periods_emax',
                        'num_periods', 'model_paras', 'states_all',
                        'num_draws_emax', 'edu_start', 'edu_max',
                        'delta', 'is_debug')

            # Auxiliary objects
            _, _, _, _, shocks_cov, shocks_cholesky = \
                dist_model_paras(model_paras, is_debug)

            # Sample draws
            draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

            # Sampling of random period and admissible state index
            period = np.random.choice(range(num_periods))
            k = np.random.choice(range(states_number_period[period]))

            # Select systematic payoffs
            payoffs_systematic = periods_payoffs_systematic[period, k, :]

            # Set up optimization task
            level = np.random.uniform(0.01, 1.00)

            args = [num_draws_emax, draws_standard, period, k,
                payoffs_systematic, edu_max, edu_start, mapping_state_idx,
                states_all, num_periods, periods_emax, is_debug, delta,
                shocks_cov, level, is_deterministic, shocks_cholesky]

            f = fort_debug.wrapper_get_payoffs_ambiguity(*args)
            py = get_payoffs_ambiguity(*args)

            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    def test_3(self):
        """ This test case compares the results from the SLSQP implementations in
        PYTHON and FORTRAN for the actual optimization problem.
        """
        # Draw random request
        maxiter = np.random.randint(1, 100)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=2)

        tiny = 1e-6

        shocks_cov = np.identity(4)*np.random.normal(size=1)**2
        level = np.random.normal(size=1)**2

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'eq'
        constraint['args'] = (shocks_cov, level)
        constraint['fun'] = divergence

        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')
        robupy_obj = solve(robupy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
            periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
            edu_max, delta, is_debug, model_paras = \
                dist_class_attributes(robupy_obj,
                    'periods_payoffs_systematic', 'states_number_period',
                    'mapping_state_idx', 'periods_emax', 'num_periods',
                    'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                    'delta', 'is_debug', 'model_paras')

        # Auxiliary objects
        _, _, _, _, _, shocks_cholesky = \
            dist_model_paras(model_paras, is_debug)

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                                np.identity(4), (num_draws_emax,))

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        args = (num_draws_emax, draws_standard, period, k, payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, num_periods,
            periods_emax, delta, shocks_cholesky)

        opt = _minimize_slsqp(criterion_ambiguity, x0, args, maxiter=maxiter,
                       ftol=ftol, constraints=constraint)

        # Stabilization. This is done as part of the fortran implementation.
        if opt['success']:
            py = opt['x']
        else:
            py = x0

        f = fort_debug.wrapper_get_worst_case(x0, maxiter, ftol, tiny,
            num_draws_emax, draws_standard, period, k, payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, num_periods,
            periods_emax, delta, is_debug, shocks_cov, level,
            shocks_cholesky)

        # Check equality. If not equal up to the tolerance, also check
        # whether the result from the FORTRAN implementation is even better.
        try:
            np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)
        except AssertionError:
            if criterion_ambiguity(f, *args) < criterion_ambiguity(py, *args):
                pass
            else:
                raise AssertionError

    def test_4(self):
        """ This test case compare the results of a debugging setup for the SLSQP
        algorithm's PYTHON and FORTRAN implementation
        """
        # Sample basic test case
        maxiter = np.random.randint(1, 100)
        num_dim = np.random.randint(2, 4)
        ftol = np.random.uniform(0.000000, 1e-5)
        x0 = np.random.normal(size=num_dim)

        # Evaluation of Rosenbrock function. We are using the FORTRAN version
        # in the development of the optimization routines.
        f90 = fort_test.wrapper_criterion_debug_function(x0, num_dim)
        py = rosen(x0)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        py = rosen_der(x0)
        f90 = fort_test.wrapper_criterion_debug_derivative(x0, len(x0))
        np.testing.assert_allclose(py, f90[:-1], rtol=1e-05, atol=1e-06)

        # Test the FORTRAN codes against the PYTHON implementation. This is
        # expected to fail sometimes due to differences in precision between the
        # two implementations. In particular, as updating steps of the optimizer
        # are very sensitive to just small differences in the derivative
        # information. The same functions are available as a FORTRAN
        # implementations.
        def debug_constraint_derivative(x):
            return np.ones(len(x))

        def debug_constraint_function(x):
            return np.sum(x) - 10.0

        # Setting up PYTHON SLSQP interface for constraints
        constraint = dict()
        constraint['type'] = 'eq'
        constraint['args'] = ()
        constraint['fun'] = debug_constraint_function
        constraint['jac'] = debug_constraint_derivative

        # Evaluate both implementations
        f = fort_test.wrapper_slsqp_debug(x0, maxiter, ftol, num_dim)
        py = _minimize_slsqp(rosen, x0, jac=rosen_der, maxiter=maxiter,
                ftol=ftol,  constraints=constraint)['x']
        np.testing.assert_allclose(py, f, rtol=1e-05, atol=1e-06)

    def test_5(self):
        """ Compare the evaluation of the criterion function for the ambiguity
        optimization and the simulated expected future value between the FORTRAN
        and PYTHON implementations. These tests are set up a separate test case
        due to the large setup cost to construct the ingredients for the interface.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['version'] = 'PYTHON'

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        # Extract class attributes
        periods_payoffs_systematic, states_number_period, mapping_state_idx, \
        periods_emax, num_periods, states_all, num_draws_emax, edu_start, \
        edu_max, delta, model_paras, is_debug = \
            dist_class_attributes(robupy_obj,
                'periods_payoffs_systematic', 'states_number_period',
                'mapping_state_idx', 'periods_emax', 'num_periods',
                'states_all', 'num_draws_emax', 'edu_start', 'edu_max',
                'delta', 'model_paras', 'is_debug')

        # Auxiliary objects
        _, _, _, _, _, shocks_cholesky = \
            dist_model_paras(model_paras, is_debug)

        # Sample draws
        draws_standard = np.random.multivariate_normal(np.zeros(4),
                            np.identity(4), (num_draws_emax,))

        shocks_mean = np.random.normal(size=2)

        # Sampling of random period and admissible state index
        period = np.random.choice(range(num_periods))
        k = np.random.choice(range(states_number_period[period]))

        # Select systematic payoffs
        payoffs_systematic = periods_payoffs_systematic[period, k, :]

        # Evaluation point
        x = np.random.random(size=2)

        # Evaluation of simulated expected future values
        py = simulate_emax(num_periods, num_draws_emax, period, k,
            draws_standard, payoffs_systematic, edu_max, edu_start,
            periods_emax, states_all, mapping_state_idx, delta,
            shocks_cholesky, shocks_mean)

        f90 = fort_debug.wrapper_simulate_emax(num_periods, num_draws_emax,
            period, k, draws_standard, payoffs_systematic, edu_max, edu_start,
            periods_emax, states_all, mapping_state_idx, delta,
            shocks_cholesky, shocks_mean)

        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Criterion function for the determination of the worst case outcomes
        args = (num_draws_emax, draws_standard, period, k, payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, num_periods,
            periods_emax, delta, shocks_cholesky)

        py = criterion_ambiguity(x, *args)
        f90 = fort_debug.wrapper_criterion_ambiguity(x, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        # Evaluation of derivative of criterion function
        tiny = np.random.uniform(0.000000, 0.5)

        py = approx_fprime(x, criterion_ambiguity, tiny, *args)
        f90 = fort_debug.wrapper_criterion_ambiguity_derivative(x, tiny, *args)
        np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

    def test_6(self):
        """ Compare results between FORTRAN and PYTHON of selected
        hand-crafted functions. In test_97() we test FORTRAN implementations
        against PYTHON intrinsic routines.
        """
        for _ in range(10):

            # Draw random request for testing purposes
            matrix = (np.random.multivariate_normal(np.zeros(4), np.identity(
                    4), 4))
            cov = np.dot(matrix, matrix.T)
            x = np.random.rand(2)
            level = np.random.random(1)
            tiny = np.random.rand()**2

            # Kullback-Leibler (KL) divergence
            py = divergence(x, cov, level)
            f90 = fort_debug.wrapper_divergence(x, cov, level)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Gradient approximation of KL divergence
            py = approx_fprime(x, divergence, tiny, cov, level)
            f90 = fort_debug.wrapper_divergence_derivative(x, cov, level, tiny)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

        for _ in range(25):

            # Create grid of admissible state space values.
            num_periods = np.random.randint(1, 15)
            edu_start = np.random.randint(1, 5)
            edu_max = edu_start + np.random.randint(1, 5)

            # Prepare interface
            min_idx = min(num_periods, (edu_max - edu_start + 1))

            # FORTRAN
            fort_a, fort_b, fort_c, fort_d = fort_lib.f2py_create_state_space(
                    num_periods, edu_start, edu_max, min_idx)

            # PYTHON
            py_a, py_b, py_c, py_d = pyth_create_state_space(num_periods,
                    edu_start, edu_max, min_idx)

            # Ensure equivalence
            for obj in [[fort_a, py_a], [fort_b, py_b], [fort_c, py_c], [fort_d, py_d]]:
                    np.testing.assert_allclose(obj[0], obj[1])

        for _ in range(100):

            # Draw random request for testing purposes
            num_covars = np.random.randint(2, 10)
            num_agents = np.random.randint(100, 1000)
            tiny = np.random.normal(size=num_agents)
            beta = np.random.normal(size=num_covars)

            # Generate sample
            exog = np.random.sample((num_agents, num_covars))
            exog[:, 0] = 1
            endog = np.dot(exog, beta) + tiny

            # Run statsmodels
            results = sm.OLS(endog, exog).fit()

            # Check parameters
            py = results.params
            f90 = fort_debug.wrapper_get_coefficients(endog, exog, num_covars,
                num_agents)
            np.testing.assert_almost_equal(py, f90)

            # Check prediction
            py = results.predict(exog)
            f90 = fort_debug.wrapper_point_predictions(exog, f90, num_agents)
            np.testing.assert_almost_equal(py, f90)

            # Check coefficient of determination
            py = results.rsquared
            f90 = fort_debug.wrapper_get_r_squared(endog, f90, num_agents)
            np.testing.assert_almost_equal(py, f90)

    def test_7(self):
        """ Compare results between FORTRAN and PYTHON of selected functions. The
        file fortran/debug_interface.f90 provides the F2PY bindings.
        """
        for _ in range(10):

            # Draw random requests for testing purposes.
            num_draws_emax = np.random.randint(2, 1000)
            dim = np.random.randint(1, 6)
            mean = np.random.uniform(-0.5, 0.5, (dim))

            matrix = (np.random.multivariate_normal(np.zeros(dim),
                np.identity(dim), dim))
            cov = np.dot(matrix, matrix.T)

            # PDF of normal distribution
            args = np.random.normal(size=3)
            args[-1] **= 2

            f90 = fort_debug.wrapper_normal_pdf(*args)
            py = norm.pdf(*args)

            np.testing.assert_almost_equal(py, f90)

            # Singular Value Decomposition
            py = scipy.linalg.svd(matrix)
            f90 = fort_debug.wrapper_svd(matrix, dim)

            for i in range(3):
                np.testing.assert_allclose(py[i], f90[i], rtol=1e-05, atol=1e-06)

            # Pseudo-Inverse
            py = np.linalg.pinv(matrix)
            f90 = fort_debug.wrapper_pinv(matrix, dim)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Inverse
            py = np.linalg.inv(cov)
            f90 = fort_debug.wrapper_inverse(cov, dim)
            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Determinant
            py = np.linalg.det(cov)
            f90 = fort_debug.wrapper_determinant(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Trace
            py = np.trace(cov)
            f90 = fort_debug.wrapper_trace(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Cholesky decomposition
            f90 = fort_debug.wrapper_cholesky(cov, dim)
            py = np.linalg.cholesky(cov)

            np.testing.assert_allclose(py, f90, rtol=1e-05, atol=1e-06)

            # Random normal deviates. This only tests the interface, requires
            # visual inspection in IPYTHON notebook as well.
            fort_debug.wrapper_standard_normal(num_draws_emax)
            fort_debug.wrapper_multivariate_normal(mean, cov, num_draws_emax,
                dim)

            # Clipping values below and above bounds.
            num_values = np.random.randint(1, 10000)
            lower_bound = np.random.randn()
            upper_bound = lower_bound + np.random.ranf()
            values = np.random.normal(size=num_values)

            f90 = fort_debug.wrapper_clip_value(values, lower_bound,
                upper_bound, num_values)
            py = np.clip(values, lower_bound, upper_bound)

            np.testing.assert_almost_equal(py, f90)

    def test_8(self):
        """ Testing ten admissible realizations of state space for the first
        three periods.
        """
        # Generate constraint periods
        constraints = dict()
        constraints['periods'] = np.random.randint(3, 5)

        # Generate random initialization file
        generate_init(constraints)

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        robupy_obj = solve(robupy_obj)

        simulate(robupy_obj)

        # Distribute class attributes
        states_number_period = robupy_obj.get_attr('states_number_period')

        states_all = robupy_obj.get_attr('states_all')

        # The next hard-coded results assume that at least two more
        # years of education are admissible.
        edu_max = robupy_obj.get_attr('edu_max')
        edu_start = robupy_obj.get_attr('edu_start')

        if edu_max - edu_start < 2:
            return

        # The number of admissible states in the first three periods
        for j, number_period in enumerate([1, 4, 13]):
            assert (states_number_period[j] == number_period)

        # The actual realizations of admissible states in period one
        assert ((states_all[0, 0, :] == [0, 0, 0, 1]).all())

        # The actual realizations of admissible states in period two
        states = [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]]
        states += [[1, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[1, j, :] == state).all())

        # The actual realizations of admissible states in period three
        states = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]
        states += [[0, 0, 2, 1], [0, 1, 0, 0], [0, 1, 1, 0]]
        states += [[0, 1, 1, 1], [0, 2, 0, 0], [1, 0, 0, 0]]
        states += [[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]]
        states += [[2, 0, 0, 0]]

        for j, state in enumerate(states):
            assert ((states_all[2, j, :] == state).all())

    def test_9(self):
        """ Testing whether back-and-forth transformation have no effect.
        """
        # Generate random request
        paras_fixed = np.random.choice([True, False], 26)

        for i in range(10):
            # Create random parameter vector
            base = np.random.uniform(size=26)
            x = base.copy()

            # Apply numerous transformations
            for j in range(10):
                args = dist_optim_paras(x, is_debug=True)
                args += ('all', paras_fixed)
                x = get_optim_paras(*args, is_debug=True)

            # Checks
            np.testing.assert_allclose(base, x)

    def test_10(self):
        """ Testing the core functions of the solution step for the equality
        of results between the PYTHON and FORTRAN implementations.
        """

        # Generate random initialization file
        generate_init()

        # Perform toolbox actions
        robupy_obj = read('test.robupy.ini')

        # Ensure that backward induction routines use the same grid for the
        # interpolation.
        write_interpolation_grid('test.robupy.ini')

        # Extract class attributes
        num_periods, edu_start, edu_max, min_idx, model_paras, num_draws_emax, \
            seed_emax, is_debug, delta, level, is_ambiguous, \
            is_interpolated, num_points, is_deterministic, is_myopic = \
                dist_class_attributes(robupy_obj,
                    'num_periods', 'edu_start', 'edu_max', 'min_idx',
                    'model_paras', 'num_draws_emax', 'seed_emax', 'is_debug',
                    'delta', 'level', 'is_ambiguous',
                    'is_interpolated', 'num_points', 'is_deterministic',
                    'is_myopic')

        # Auxiliary objects
        coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cov, \
            shocks_cholesky = dist_model_paras(model_paras, is_debug)

        # Check the state space creation.
        args = (num_periods, edu_start, edu_max, min_idx)
        pyth = pyth_create_state_space(*args)
        f2py = f2py_create_state_space(*args)
        for i in range(4):
            np.testing.assert_allclose(pyth[i], f2py[i])

        # Carry some results from the state space creation for future use.
        states_all, states_number_period = pyth[:2]
        mapping_state_idx, max_states_period = pyth[2:]

        # Check calculation of systematic components of payoffs.
        args = (num_periods, states_number_period, states_all, edu_start,
            coeffs_a, coeffs_b, coeffs_edu, coeffs_home, max_states_period)
        pyth = pyth_calculate_payoffs_systematic(*args)
        f2py = f2py_calculate_payoffs_systematic(*args)
        np.testing.assert_allclose(pyth, f2py)

        # Carry some results from the systematic payoff calculation for
        # future use and create the required set of disturbances.
        periods_draws_emax = create_draws(num_periods, num_draws_emax, seed_emax,
            is_debug)

        periods_payoffs_systematic = pyth

        # Check backward induction procedure.
        args = (num_periods, max_states_period, periods_draws_emax,
            num_draws_emax, states_number_period, periods_payoffs_systematic,
            edu_max, edu_start, mapping_state_idx, states_all, delta,
            is_debug, shocks_cov, level, is_ambiguous,
            is_interpolated, num_points, is_deterministic, shocks_cholesky)
        pyth = pyth_backward_induction(*args)
        f2py = f2py_backward_induction(*args)
        np.testing.assert_allclose(pyth, f2py)
