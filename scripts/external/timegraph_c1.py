"""TimeGraph C1 generator, vendored for the external-family experiment.

Source: Codes/c1.py of the TimeGraph benchmark (Ferdous, Hossain, and Gani,
KDD 2025, https://github.com/hferdous/TimeGraph, commit 4d93403b, June 2025).
Only notebook scaffolding was removed (a pip install line, plotting imports,
and the dataset-saving helpers). The equations and the generator class below
are identical to the upstream file. Used by scripts/run_timegraph_c1.py.

Upstream license (MIT):

    Copyright (c) 2025 Muhammad Hasan Ferdous

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def get_nonlinear_equations(n_vars, max_lag):
    """Get nonlinear equations following A1's exact causal structure including lags"""
    if n_vars == 4:
        if max_lag == 2:
            return [
                # Following A1's structure exactly with nonlinear functions and same lags
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + trend3[t] + season3[t] + e3",  # contemporaneous
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",  # lag-1
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"  # contemporaneous
            ]
        elif max_lag == 3:
            return [
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
        elif max_lag == 4:
            return [
                "X4[t] = 0.25 * cos(X1[t-4] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
    elif n_vars == 6:
        if max_lag == 2:
            return [
                # Following A1's structure for 6 variables with exact lags
                "X6[t] = 0.85 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",  # contemporaneous
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",  # lag-1
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + 0.3 * sin(X5[t-1] * pi/2) + trend4[t] + season4[t] + e4",  # lag-2, lag-1
                "X3[t] = 0.35 * (X4[t])^2 + trend3[t] + season3[t] + e3",  # contemporaneous
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",  # lag-1
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"  # contemporaneous
            ]
        elif max_lag == 3:
            return [
                "X6[t] = 0.85 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + trend4[t] + season4[t] + e4",  # Removed X5[t-1] term
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",  # Additional lag-3 term
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
        elif max_lag == 4:
            return [
                "X6[t] = 0.85 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",
                "X4[t] = 0.25 * cos(X1[t-4] * pi/2) + trend4[t] + season4[t] + e4",  # Changed to lag-4, removed X5[t-1] term
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
    elif n_vars == 8:
        if max_lag == 2:
            return [
                # Following A1's structure for 8 variables with exact lags
                "X8[t] = 0.4 * sin(X7[t] * pi/2) + trend8[t] + season8[t] + e8",
                "X7[t] = 0.35 * cos(X6[t-1] * pi/2) + trend7[t] + season7[t] + e7",
                "X6[t] = 0.45 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + 0.3 * sin(X5[t-1] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
        elif max_lag == 3:
            return [
                "X8[t] = 0.4 * sin(X7[t] * pi/2) + trend8[t] + season8[t] + e8",
                "X7[t] = 0.35 * cos(X6[t-1] * pi/2) + trend7[t] + season7[t] + e7",
                "X6[t] = 0.45 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",
                "X4[t] = 0.25 * cos(X1[t-2] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]
        elif max_lag == 4:
            return [
                "X8[t] = 0.4 * sin(X7[t] * pi/2) + trend8[t] + season8[t] + e8",
                "X7[t] = 0.35 * cos(X6[t-1] * pi/2) + trend7[t] + season7[t] + e7",
                "X6[t] = 0.45 * sin(X5[t] * pi/2) + trend6[t] + season6[t] + e6",
                "X5[t] = 0.4 * cos(X4[t-1] * pi/2) + trend5[t] + season5[t] + e5",
                "X4[t] = 0.25 * cos(X1[t-4] * pi/2) + trend4[t] + season4[t] + e4",
                "X3[t] = 0.35 * (X4[t])^2 + 0.2 * cos(X2[t-3] * pi/2) + trend3[t] + season3[t] + e3",
                "X2[t] = 0.3 * sin(X3[t-1] * pi/2) + trend2[t] + season2[t] + e2",
                "X1[t] = 0.4 * (X2[t])^3 + trend1[t] + season1[t] + e1"
            ]

    return []

class NonlinearTimeSeriesGenerator:
    def __init__(self, noise_scale=0.1, trend_strength=0.01, seasonal_strength=0.5,
                 seasonal_period=12, random_state=None):
        self.noise_scale = noise_scale
        self.trend_strength = trend_strength
        self.seasonal_strength = seasonal_strength
        self.seasonal_period = seasonal_period
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

    def generate_noise(self, size):
        """Generate Gaussian noise"""
        return np.random.normal(0, self.noise_scale, size=size)

    def generate_trend(self, n_points, var_idx):
        """Generate deterministic trend component"""
        trend_modifier = (var_idx + 1) * 0.5
        t = np.arange(n_points)
        return self.trend_strength * trend_modifier * t

    def generate_seasonality(self, n_points, var_idx):
        """Generate seasonal component using sine and cosine terms"""
        phase_shift = 2 * np.pi * var_idx / 8
        t = np.arange(n_points)

        # Combine multiple seasonal frequencies
        season1 = np.sin(2 * np.pi * t / self.seasonal_period + phase_shift)
        season2 = 0.5 * np.cos(4 * np.pi * t / self.seasonal_period + phase_shift)

        return self.seasonal_strength * (season1 + season2)

    def evaluate_term(self, term, var_values, X, t):
        """Evaluate a single term in the equation"""
        parts = term.split('*')
        coef = float(parts[0].strip())
        expr = parts[1].strip()

        # Handle autocorrelation terms
        if '[t-' in expr and not any(func in expr for func in ['cos', 'sin', '^']):
            var_idx = int(expr[1]) - 1  # Extract number after 'X'
            lag = int(expr.split('-')[1].split(']')[0])
            return coef * X[t-lag, var_idx] if t > lag else 0

        # Handle nonlinear terms
        if 'cos(' in expr:
            inner = expr.split('cos(')[1].split(')')[0]
            var_idx = int(inner.split('X')[1].split('[')[0]) - 1
            time_part = inner.split('[')[1].split(']')[0]
        elif 'sin(' in expr:
            inner = expr.split('sin(')[1].split(')')[0]
            var_idx = int(inner.split('X')[1].split('[')[0]) - 1
            time_part = inner.split('[')[1].split(']')[0]
        elif '^' in expr:
            power = int(expr.split('^')[1])
            var_idx = int(expr.split('X')[1].split('[')[0]) - 1
            time_part = expr.split('[')[1].split(']')[0]
        else:
            var_idx = int(expr.split('X')[1].split('[')[0]) - 1
            time_part = expr.split('[')[1].split(']')[0]

        # Get the value based on time index
        if time_part == 't':
            var_name = f'X{var_idx + 1}'
            value = var_values.get(var_name, X[t, var_idx])
        else:
            lag = int(time_part.split('-')[1])
            value = X[t-lag, var_idx] if t >= lag else 0

        # Apply the appropriate transformation
        if 'cos(' in expr:
            return coef * np.cos(value * np.pi/2)
        elif 'sin(' in expr:
            return coef * np.sin(value * np.pi/2)
        elif '^' in expr:
            return coef * (value ** power)
        return coef * value

    def generate_equations(self, t, X, trends, seasonality, n_vars, max_lag):
        """Execute nonlinear equations with trends and seasonality"""
        noise = self.generate_noise(n_vars)
        equations = get_nonlinear_equations(n_vars, max_lag)
        var_values = {}

        # Process each equation
        for eq in equations:
            if '=' not in eq:
                continue

            left, right = eq.split('=')
            var_name = left.split('[')[0]
            var_idx = int(var_name[1:]) - 1

            terms = right.split('+')
            value = 0

            for term in terms:
                term = term.strip()
                if term.startswith('e'):  # Noise term
                    value += noise[var_idx]
                elif term.startswith('trend'):
                    value += trends[var_idx][t]
                elif term.startswith('season'):
                    value += seasonality[var_idx][t]
                else:
                    value += self.evaluate_term(term, var_values, X, t)

            X[t, var_idx] = value
            var_values[var_name] = value

    def generate_multivariate_ts(self, n_points, n_vars, max_lag):
        """Generate multivariate time series with trends and seasonality"""
        X = np.zeros((n_points, n_vars))

        # Generate trends and seasonality for each variable
        trends = [self.generate_trend(n_points, i) for i in range(n_vars)]
        seasonality = [self.generate_seasonality(n_points, i) for i in range(n_vars)]

        # Initialize first steps with noise
        for i in range(max_lag):
            X[i] = self.generate_noise(n_vars)
            # Add trend and seasonality to initial values
            for j in range(n_vars):
                X[i, j] += trends[j][i] + seasonality[j][i]

        # Generate time series
        for t in range(max_lag, n_points):
            self.generate_equations(t, X, trends, seasonality, n_vars, max_lag)

        # Create DataFrame
        timestamps = np.arange(n_points)
        columns = [f'X{i+1}' for i in range(n_vars)]
        df = pd.DataFrame(X, columns=columns)
        df['time'] = timestamps

        return df
