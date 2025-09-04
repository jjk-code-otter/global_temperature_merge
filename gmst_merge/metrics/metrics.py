#  Global Temperature Merge - a package for merging global temperature datasets.
#  Copyright \(c\) 2025 John Kennedy and Bruce Calvert
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import statsmodels.api as sm

def calculate_ipcc_long_term_change(intime, inarray, label=False, range=None):
    if label:
        return 'IPCC AR6 difference 2001-2020 minus 1850-1900'

    if range is not None:
        return [0.05, 0.95]

    early_period = (1850 <= intime) & (intime <= 1900)
    late_period = (2001 <= intime) & (intime <= 2020)

    out_value = np.mean(inarray[late_period]) - np.mean(inarray[early_period])

    return out_value


def calculate_craigmile_and_guttorp(intime, inarray, label=False, range=False):
    if label:
        return 'Craigmile and Guttorp 1995-2014 minus 1880-1899'

    if range:
        return [0.025, 0.975]

    early_period = (1880 <= intime) & (intime <= 1899)
    late_period = (1995 <= intime) & (intime <= 2014)

    out_value = np.mean(inarray[late_period]) - np.mean(inarray[early_period])

    return out_value


def calculate_2024_over_15C(intime, inarray, label=False, range=False):
    if label:
        return 'Probability that 2024 was 1.5C or above'

    if range:
        return [0, 0]

    selected = (intime == 2024)

    return bool((inarray[selected] >= 1.5)[0])

def calculate_2023_over_15C(intime, inarray, label=False, range=False):
    if label:
        return 'Probability that 2023 was 1.5C or above'

    if range:
        return [0, 0]

    selected = (intime == 2023)

    return bool((inarray[selected] >= 1.5)[0])


def calculate_smoothed_warming_in_last_year(intime, inarray, label=False, range=False):
    if label:
        return f'Long-term warming in {intime[-1]}'

    if range:
        return [0.025, 0.975]

    lowess = sm.nonparametric.lowess
    z = lowess(inarray, intime, frac=1. / 5., return_sorted=False)

    return z[-1]

def calculate_prob_smoothed_warming_in_last_year_over15C(intime, inarray, label=False, range=False):
    if label:
        return f'Probability long-term warming in {intime[-1]} > 1.5C'

    if range:
        return [0, 0]

    lowess = sm.nonparametric.lowess
    z = lowess(inarray, intime, frac=1. / 5., return_sorted=False)

    return bool(z[-1] >= 1.5)