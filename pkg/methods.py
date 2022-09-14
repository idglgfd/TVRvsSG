"""
you need to install cvxpy and Mosek
conda install -c conda-forge cvxpy
conda install -c mosek mosek
"""

import numpy as np
import cvxpy
import jackknife as jk
from scipy import interpolate
from scipy.stats import linregress


def total_variation_regularized_derivative(x, t, N, gamma, solver='MOSEK', mode=0):
    """
    Use convex optimization (cvxpy) to solve for the Nth total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    :param x: (np.array of floats, 1xN) time series to differentiate
    :param t: (np.array of floats, 1xN) time
    :param N: (int) 1, 2, or 3, the Nth derivative to regularize
    :param gamma: (float) regularization parameter
    :param solver: (string) Solver to use. Solver options include: 'MOSEK' and 'CVXOPT',
                            in testing, 'MOSEK' was the most robust.
    :param mode: if 0 - uses sum of |dx/dt| elif 1 - uses sum of |dx/dt * dt|
    :return: x_hat    : estimated (smoothed) x
             dxdt_hat : estimated derivative of x
    """

    # normalize
    mean = np.mean(x)
    std = np.std(x)
    x = (x - mean) / std

    dt = np.zeros_like(t)  # zero in ii==0 !!!
    dt[1:] = t[1:] - t[:-1]

    # Define the variables for the highest order derivative and the integration constants
    var = cvxpy.Variable(len(x) + N)

    # Recursively integrate the highest order derivative to get back to the position
    derivatives = [var[N:]]
    for i in range(N):
        d = cvxpy.cumsum(cvxpy.multiply(derivatives[-1], dt[:])) + var[i]
        derivatives.append(d)

    # Compare the recursively integration position to the noisy position
    sum_squared_error = cvxpy.sum_squares(derivatives[-1] - x)

    # Total variation regularization on the highest order derivative
    if mode == 0:
        r = cvxpy.sum(gamma * cvxpy.tv(derivatives[0]))  # sum of |dx/dt|
    elif mode == 1:
        r = cvxpy.sum(gamma * cvxpy.tv(cvxpy.multiply(derivatives[0], dt[:])))  # sum of |dx/dt * dt|
    else:
        raise FutureWarning('Wrong mode')

    # Set up and solve the optimization problem
    obj = cvxpy.Minimize(sum_squared_error + r)
    prob = cvxpy.Problem(obj)
    prob.solve(solver=solver)

    # Recursively calculate the value of each derivative
    final_derivative = var.value[N:]
    derivative_values = [final_derivative]
    for i in range(N):
        d = np.cumsum(derivative_values[-1] * dt[:]) + var.value[i]
        derivative_values.append(d)

    # Extract the velocity and smoothed position
    dxdt_hat = derivative_values[-2]
    x_hat = derivative_values[-1]

    return x_hat * std + mean, dxdt_hat * std


def jackknife_tvr(x, t, N=2, gamma=1, mode=1):
    """
    Uses tvr and jackknife lib to both estimate fit and derivative and jackknife stats
    :param x: (np.array of floats, 1xN) time series to differentiate
    :param t: (np.array of floats, 1xN) time
    :param N: (int) 1, 2, or 3, the Nth derivative to regularize
    :param gamma: (float) regularization parameter
    :param mode: if 0 - uses sum of |dx/dt| elif 1 - uses sum of |dx/dt * dt|
    :return: fit, fit_stats, dxdt, dxdt_stats
    """

    data = np.stack((t, x), axis=-1)  # data to matrix

    def fit_it(data):
        """fit procedure"""
        x_fit, _ = total_variation_regularized_derivative(data[:, 1], data[:, 0], N, gamma=gamma, mode=mode)
        return x_fit

    def dxdt_it(data):
        """deriv procedure"""
        _, dxdt = total_variation_regularized_derivative(data[:, 1], data[:, 0], N, gamma=gamma, mode=mode)
        return dxdt

    def jk_it(proc, data):
        """jackknife estimation and stats with some procedure"""
        theta_subsample, theta_fullsample = jk.jk_loop(proc, data)  # jackknife data

        '''adding missing predictions to subsample matrix by interpolating'''
        theta_subsample_ext = np.zeros(
            (theta_subsample.shape[0], theta_subsample.shape[1] + 1))  # to add diag elements with predictions
        for ii, sub in enumerate(theta_subsample):
            xxx = data[:, 0].copy()
            x, xx = xxx[ii], np.delete(xxx, ii)
            f = interpolate.interp1d(xx, sub, bounds_error=False, fill_value="extrapolate")
            y = f(x)
            theta_subsample_ext[ii] = np.insert(sub, ii, y)

        _, _, theta_jack, se_jack, theta_biased = jk.jk_stats(theta_subsample_ext, theta_fullsample)
        stats = {'theta_jack': theta_jack,
                 'se_jack': se_jack,
                 'theta_biased': theta_biased}

        return theta_fullsample, stats

    fit, fit_stats = jk_it(fit_it, data)
    dxdt, dxdt_stats = jk_it(dxdt_it, data)

    return fit, fit_stats, dxdt, dxdt_stats


def sav_gol_deriv(x, t, window=7):
    """
    Savicky-Golay 1st order smoothing and derivative calculation
    :param x:
    :param t:
    :param window: smooting window size
    :return: x_fit, fit_se, deriv, deriv_se
    """

    N = len(x)  # end of arr
    x_fit = np.zeros_like(x)
    deriv = np.zeros_like(x)
    fit_se = np.zeros_like(x)
    deriv_se = np.zeros_like(x)
    range = np.ones_like(x) * window  # just for convenience

    for ii, r in enumerate(range):
        hf = max(r // 2, 1)  # half of range or 1
        low = int(max(ii - hf, 0))
        hi = int(min(ii + hf + 1, N))

        dt = t[low:hi]  # piece of t
        dx = x[low:hi]

        res = linregress(dt, dx)
        deriv[ii] = res.slope
        deriv_se[ii] = res.stderr

        dx_fit = res.slope * dt + res.intercept
        x_fit[ii] = res.slope * t[ii] + res.intercept

        dof = len(dt) - 2
        fit_se = np.sqrt(np.sum((dx - dx_fit) ** 2) / dof)

    return x_fit, fit_se, deriv, deriv_se
