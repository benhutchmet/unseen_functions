#!/usr/bin/env python
"""Bias Adjustment for Python

Designed for compatibility with ArrayLike data types, including numpy array,
pandas series, xarray data array. Bias adjustments are a method of adjusting one
dataset relative to another, to allow comparison. For example, a global climate
model may exhibit biases in the historic period, however, correcting for this
bias allows an improved assessment of present and future climate risk.

Different bias adjustments are appropriate when used for different purposes, on
different sample rates and sample sizes, for different variables, etc.

Mean bias adjustment ('linear_scaling') is the simplest form of bias correction,
making an adjustment for differences in the mean value. More complicated
adjustments based on higher order moments (variance, skew, etc.), on quantiles,
or otherwise.

Takes inspiration from climQMBC and bias-correction modules. The new "Bias
Adjustment" module aims to be as simple as possible to use, whilst including
complexity required for complex bias adjustments applied to arrays whilst
preserving metadata.

Notes:
======
    Available methods: 
        MeanBiasAdjust:
            Simple linear scaling method that only adjusts for mean
            bias. Applies constant offset.
        VarBiasAdjust:
            Simple method that adjusts both mean and variance bias.
            Applies constant offset to mean, scaling factor variance.
        QMBiasAdjust:
            Quantile Mapping, preserves distribution. See Themeßl et al. (2011)
            https://doi.org/10.1002/joc.2168 and Cannon et al. (2015)
            https://doi.org/10.1175/JCLI-D-14-00754.1
        QDMBiasAdjust:
            Quantile Delta Mapping, used to minimise distributional biases
            between observations and model. See Tong et al. (2021)
            https://doi.org/10.1007/s00382-020-05447-4
        SDMBiasAdjsut:
            Scaled distribution mapping adjustment, preserves distribution.
            See Switanek et al. 2017 https://doi.org/10.5194/hess-21-2649-2017


Alternatives:
=============
    This bias adjust python module takes inspiration from
    [climQMBC](https://github.com/saedoquililongo/climQMBC),
    [bias-correction](https://github.com/pankajkarman/bias_correction), and
    [python-cmethods](https://pypi.org/project/python-cmethods/) and
    [ibicus](https://github.com/ecmwf-projects/ibicus).



Information:
============
    Author: James Fallon
    Email: online@jamesfallon.eu
    Date: 2023-07-17, Last Updated 2023-08-25
"""

__author__ = "James Fallon"
__credits__ = ["James Fallon"]
__license__ = "GPL"
__version__ = "0.0.5"
__maintainer__ = "James Fallon"
__email__ = "j.fallon@pgr.reading.ac.uk"
__status__ = "Development"

import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.interpolate
from statsmodels.distributions.empirical_distribution import ECDF

from typing import Callable, Literal, Optional, Union
from numpy.typing import ArrayLike


def calculate_statistics(
        data: pd.Series,
        stats_index: ArrayLike
    ) -> pd.DataFrame:
    """Calculate statistics of datasets in data

    Parameters:
    ===========
    data : ArrayLike
        Input dataseries
    stats_index : ArrayLike
        List of operations to perform {"mean", "std", "skew"}

    Returns:
    ========
    stats : pd.DataFrame
        Requested statistics

    Usage:
    ======
    >>> stats = calculate_statistics(data, ["mean", "std"])
    """
    stats = pd.DataFrame(index=data.index)
    if "mean" in stats_index:
        stats["mean"] = data.apply(np.nanmean, convert_dtype=float)
    if "std" in stats_index:
        stats["std"] = data.apply(np.nanstd, convert_dtype=float)
    if "skew" in stats_index:
        stats["skew"] = data.apply(scipy.stats.skew, convert_dtype=float)
    return stats


def get_stats(
        stat: str,
        data: Optional[ArrayLike],
        stats: Optional[pd.Series]
    ) -> float:
    """Get requested statistics

    First check if available in pre-processed stats, and fallback on new
    calcualation on data. 

    Parameters:
    ===========
    stat : str
        Requested statistics {"mean", "std", "skew"}
    data : Optional[ArrayLike]
        Data from which statistics can be calculated
    stats : Optional[pd.Series]
        Series of statistics pre-calculated for data

    Returns:
    ========
    requested_stat : float
        The requested statistic

    Usage:
    ======
    >>> mean = get_stats("mean", data)
    """
    # use pre-calculated stat?
    if stats is not None:
        if stat in stats:
            return stats.get(stat)
    elif data is None:
        raise ValueError("Must provide at least one of data or stats!")
    # calculate requested statistic
    if stat == "mean":
        return np.nanmean(data)
    if stat == "std":
        return np.nanstd(data)
    if stat == "skew":
        return scipy.stats.skew(data)
    raise RuntimeError("Could not get requested statistic!")


def fit_cdf(
        distribution: Literal["empirical", "normal"],
        data: Optional[ArrayLike]=None,
        stats: Optional[pd.Series]=None
    ) -> Callable[[ArrayLike], ArrayLike]:
    """Fit cumulative distribution function

    Parameters:
    ===========
    distribution : Literal["empirical", "normal"],
        Specify the underlying probability distribution function 
    data : Optional[ArrayLike]=None,
        Data to use in calculating the cdf
    stats : Optional[pd.Series]=None
        Pre-processed stats of data, to use in calculating the cdf

    Returns:
    ========
    cdf : Callable[[ArrayLike], ArrayLike] 
        Requested cumulative distribution function, takes input timeseries

    Usage:
    ======
    >>> ecdf = fit_cdf("empirical", data)
    >>> ecdf([5.01, -2.15, 3.98, 24.23])
    array([0.32, 0.07, 0.25, 0.999994])
    """
    if distribution == "empirical":
        if data is None:
            raise ValueError("Empirical CDF calculation requires data!")
        return ECDF(data)
    elif distribution == "normal":
        mean = get_stats("mean", data, stats)
        std = get_stats("std", data, stats)
        return lambda x: scipy.stats.norm.cdf(x, mean, std)
    else:
        raise ValueError("Requested cdf type not found!")


def threshold_cdf(
    cdf_data: ArrayLike,
    lower_bound: float,
    upper_bound: Optional[float]=None
    ) -> ArrayLike:
    """Apply thresholds to cdf data (to avoid subsequent possible infinities)

    Parameters:
    ===========
    cdf_data : ArrayLike,
        cumulative distribution data
    lower_bound : float,
        lower probability cutoff
    upper_bound : Optional[float]=None
        upper probability cutoff (default = 1 - lower_bound)

    Returns:
    ========
    thresholded_cdf_data : ArrayLike
        cdf_data with upper and lower threshold bounds applied

    Usage:
    ======
    >>> threshold(ecdf([5.01, -2.15, 3.98, 24.23]))
    array([0.32, 0.07, 0.25, 0.999994])
    >>> threshold(ecdf([5.01, -2.15, 3.98, 24.23]), 1e-3)
    array([0.32, 0.07, 0.25, 0.999])
    """
    upper_bound = 1 - lower_bound if upper_bound is None else upper_bound
    return np.maximum(np.minimum(cdf_data, upper_bound), lower_bound)


def fit_inv_cdf(
        distribution: Literal["empirical", "normal"],
        data: Optional[ArrayLike]=None,
        stats: Optional[pd.Series]=None,
        cdf: Optional[Callable[[ArrayLike], ArrayLike]]=None
    ) -> Callable[[ArrayLike], ArrayLike]:
    """Fit inverse cumulative distribution function

    Parameters:
    ===========
    distribution : Literal["empirical", "normal"],
        Specify the underlying probability distribution function 
    data : Optional[ArrayLike]=None,
        Data to use in calculating the inverse cdf
    stats : Optional[pd.Series]=None
        Pre-processed stats of data, to use in calculating the inverse cdf
    cdf : Optional[Callable[[ArrayLike], ArrayLike]]
        Cumulative distribution function of data 

    Returns:
    ========
    inv_cdf : Callable[[ArrayLike], ArrayLike] 
        Requested inverse cumulative distribution function

    Usage:
    ======
    >>> inv_ecdf = fit_inv_cdf("empirical", data, cdf=cdf)
    >>> inv_ecdf([0, 0.25, 0.5, 0.75, 1.0])
    array([-3.87890635, -0.6425686 , -0.00528555,  0.71383866,  inf])
    """
    if distribution == "empirical":
        if data is None or cdf is None:
            raise ValueError("Empirical CDF calculation requires data and cdf!")
        # get interpolating inverted cdf function
        slope_changes = np.sort(data)
        inverted_cdf = scipy.interpolate.interp1d(
            cdf(slope_changes),
            slope_changes,
            bounds_error=False
        )
        # set nan values outside the resolution range (probability 1/len)
        #kwargs = dict(bounds_error=False, fill_value=(-np.inf, +np.inf))
        #inverted_cdf = scipy.interpolate.interp1d(cdf(slope_changes), slope_changes, **kwargs)
        #res = 1 / np.size(data)
        #lower_bound = res
        #upper_bound = 1 - res
        #return lambda x: np.where((lower_bound < x) * (upper_bound > x), inverted_cdf(x), np.nan)
        return inverted_cdf
    elif distribution == "normal":
        mean = get_stats("mean", data, stats)
        std = get_stats("std", data, stats)
        return lambda x: scipy.stats.norm.ppf(x, mean, std)
    else:
        raise ValueError("Requested inverse cdf type not found!")


def rolling_mean(arr: ArrayLike, window: Union[float, int]=0.005) -> np.ndarray:
    # if window is a float, convert to int (multiply by array length)
    arr_len = len(arr)
    if isinstance(window, float):
        assert 0 < window < 1, "invalid value for smoothing_window"
        window = int(arr_len * window)
    # check window is a valid type
    assert isinstance(window, int), "invalid type for smoothing_window"
    assert 0 < window < arr_len, "invalid window size"
    # calculate rolling mean
    rolling_kwargs = dict(center=True, min_periods=0, closed="neither")
    return pd.Series(arr).rolling(window, **rolling_kwargs).mean().values


class BiasAdjust:
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        scaling_type: Optional[Literal["additive", "relative"]]=None
        ):
        """Bias Adjustment Class

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)
        scaling_type : Optional[Literal["additive", "relative"]]
            Is bias adjustment additive (like temperature) or relative
            (like precipitation)? Should non-negative values be enforced?
        """
        # Initialise parameters
        self.scaling_type = scaling_type

        # Construct data Series
        index = ["obs", "mod"]
        self.data = pd.Series([obs_data, mod_data], index=index)

        # Calculate basic statistics
        self.stats = calculate_statistics(self.data, ["mean", "std"])

        # initialise empty parameters that are calculated later
        self.input = None
        self.corrected = None

    def __repr__(self):
        shape = map(np.shape, self.data)
        return "BiasAdjust(obs_data=<{}>, mod_Data=<{}>)".format(*shape)
    
    def __str__(self):
        str_out = [
            f"Observations data: {self.data['obs']}",
            f"Model data: {self.data['mod']}",
            f"Stats:\n{self.stats}"
        ]
        if self.input is not None:
            str_out.append(f"Input: {self.input}")
        if self.corrected is not None:
            str_out.append(f"Corrected: {self.corrected}")
        return "\n\n".join(str_out)

    def floor_corrected(self):
        if self.corrected is not None:
            self.corrected = np.clip(self.corrected, 0, np.inf)
        return self.corrected
      
    def _correct(self, bias_adjustment: Callable[[ArrayLike], ArrayLike], input: Optional[ArrayLike]=None) -> ArrayLike:
        # store input timeseries
        self.input = input
        # if no input timeseries provided, correct the original model data
        if input is None:
            self.input = self.data["mod"]
        # apply correction and store in corrected
        self.corrected = bias_adjustment()
        # return corrected timeseries
        return self.corrected


class MeanBiasAdjust(BiasAdjust):
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        scaling_type: Literal["additive", "relative"]="additive"
        ):
        """Mean Bias Adjustment Class

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)
        scaling_type : Literal["additive", "relative"]
            Is mean bias adjustment additive (like temperature) or relative
            (like precipitation)? (default additive)

        Usage:
        ======
        >>> adjustment = MeanBiasAdjust(obs, rcm_hist)
        >>> rcm_hist_corrected = adjustment.correct()
        >>> rcm_futr_corrected = adjustment.correct(rcm_futr)
        """
        super().__init__(obs_data, mod_data, scaling_type)

    def linear_adjust(self) -> ArrayLike:
        mean = self.stats["mean"]
        return self.input - (mean["mod"] - mean["obs"])

    def linear_rel_adjust(self) -> ArrayLike:
        mean = self.stats["mean"]
        return mean["obs"] / mean["mod"] * self.input

    def correct(self, input: Optional[ArrayLike]=None) -> ArrayLike:
        # select appropriate method
        method = {
            "additive": self.linear_adjust,
            "relative": self.linear_rel_adjust
        }[self.scaling_type]
        # apply correction
        return super()._correct(method, input)


class VarBiasAdjust(BiasAdjust):
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        ):
        """Variance Scaling Bias Adjustment Class

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)

        Usage:
        ======
        >>> adjustment = VarBiasAdjust(obs, rcm_hist)
        >>> rcm_hist_corrected = adjustment.correct()
        >>> rcm_futr_corrected = adjustment.correct(rcm_futr)
        """
        super().__init__(obs_data, mod_data)

    def variance_adjust(self) -> ArrayLike:
        mean, std = self.stats["mean"], self.stats["std"]
        return std["mod"] / std["obs"] * (self.input - mean["mod"]) + mean["obs"]

    def correct(self, input: Optional[ArrayLike]=None) -> ArrayLike:
        return super()._correct(self.variance_adjust, input)


class QMBiasAdjust(BiasAdjust):
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        distribution: Literal["empirical", "normal"]="empirical"
        ):
        """Quantile Mapping Bias Adjustment Class

        1. Calculate the empirical CDF of model data (historic period).
        2. Calculate the inverse empirical CDF of observations data.
        3. Apply the CDF to input data (model future/alt period).
        4. Calculate obs_invECDF(mod_ECDF(input))

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)
        distribution : Literal["empirical", "normal"]
            Distribution used in fitting the CDF and CDF-inverse

        Usage:
        ======
        >>> adjustment = QMBiasAdjust(obs, rcm_hist)
        >>> rcm_hist_corrected = adjustment.correct()
        >>> rcm_futr_corrected = adjustment.correct(rcm_futr)
        """
        super().__init__(obs_data, mod_data)
        self.distribution = distribution

        # calculate the cdf, inverse cdf of modelled data
        self.cdf = pd.Series(index=self.data.index, dtype=object)
        self.inv_cdf = pd.Series(index=self.data.index, dtype=object)
        for key in self.data.index:
            self.cdf[key] = fit_cdf(distribution, self.data[key], self.stats.loc[key])
            self.inv_cdf[key] = fit_inv_cdf(distribution, self.data[key], cdf=self.cdf[key])

    def qm_adjust(self) -> ArrayLike:
        # apply model data cdf to input data
        tao = self.cdf["mod"](self.input)

        print(f"tao qm adjust: {tao}")

        # calculate the inverse cdf (model into observations space)
        xvals = self.inv_cdf["obs"](tao)

        print("Xvals qm adjust", xvals)
        # preserve original format (pandas or xarray)
        out = self.input * 0

        # print out
        print("Out from qm adjust", out)

        # print out and xvals
        print("Out + xvals", out + xvals)

        return out + xvals

    def correct(self, input: Optional[ArrayLike]=None) -> ArrayLike:
        return super()._correct(self.qm_adjust, input)


class QDMBiasAdjust(BiasAdjust):
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        distribution: Literal["empirical", "normal"]="empirical",
        smoothing: bool=False,
        smoothing_window: Union[int, float]=0.005
        ):
        """Quantile Delta Mapping Bias Adjustment Class

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)
        distribution : Literal["empirical", "normal"]
            Distribution used in fitting the CDF and CDF-inverse
        smoothing : bool
            Use mean rolling window on the "delta" (helps prevent
            non-monotonicity in < 0.005 and > 0.995 percentiles).
        smoothing_window : Optional[int, float]
            Window size, either directly specified (int) or to be calculated as
            a fraction of array length (float).

        Usage:
        ======
        >>> adjustment = QDMBiasAdjust(obs, rcm_hist)
        >>> rcm_hist_corrected = adjustment.correct()
        >>> rcm_futr_corrected = adjustment.correct(rcm_futr)
        """
        super().__init__(obs_data, mod_data)
        self.distribution = distribution

        # calculate the cdf, inverse cdf of modelled data
        self.cdf = pd.Series(index=self.data.index, dtype=object)
        self.inv_cdf = pd.Series(index=self.data.index, dtype=object)
        for key in self.data.index:
            self.cdf[key] = fit_cdf(distribution, self.data[key], self.stats.loc[key])
            self.inv_cdf[key] = fit_inv_cdf(distribution, self.data[key], cdf=self.cdf[key])
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window

    def qdm_adjust(self) -> ArrayLike:
        print("QDM Adjust")
        # apply model future data cdf to model future data
        cdf_input = fit_cdf(self.distribution, self.input)
        tao = cdf_input(self.input)

        print(f"tao qdm adjust: {tao}")
        # calculate the inverse cdf (model into observations space)
        xvals = self.inv_cdf["obs"](tao)

        print("Xvals qdm adjust", xvals)

        # Absolute change in quantiles between model historic / future
        yvals = self.inv_cdf["mod"](tao)
        
        # Previously calculated change added to the bias-adjusted values
        delta = xvals - yvals

        print("Delta", delta)

        # ptrin the type of self.input
        print(type(self.input))

        # print the values of self.input + delta
        print(f"self.input + delta: {self.input + delta}")

        return self.input + delta

    def qdsm_adjust(self) -> ArrayLike:
        print("QDSM Adjust")
        # quantile delta-smoothed mapping (requires sorted input array)
        # apply model future data cdf to model future data
        cdf_input = fit_cdf(self.distribution, self.input)
        z_idx = np.argsort(self.input)
        z = np.asarray(self.input)[z_idx]
        tao = cdf_input(z)
        # calculate the inverse cdf (model into observations space)
        xvals = self.inv_cdf["obs"](tao)
        # Absolute change in quantiles between model historic / future
        yvals = self.inv_cdf["mod"](tao)
        # Previously calculated change added to the bias-adjusted values
        # with rolling mean
        delta = rolling_mean(xvals - yvals, self.smoothing_window)
        # apply correction to sorted data
        corrected_sorted = z + delta
        # reverse the sorting
        return self.input*0 + corrected_sorted[np.argsort(z_idx)]

    def correct(self, input: Optional[ArrayLike]=None) -> ArrayLike:
        method = [self.qdm_adjust, self.qdsm_adjust][self.smoothing]
        return super()._correct(method, input)


class SDMBiasAdjust(BiasAdjust):
    def __init__(
        self,
        obs_data: ArrayLike,
        mod_data: ArrayLike,
        distribution: Literal["normal", "gamma"]="normal",
        cdf_threshold: float=1e-5,
        detrend_type: str="constant"
        ):
        """Scaled Distribution Mapping Bias Adjustment Class

        Scaled distribution mapping based on Switanek et al. 2017
        https://doi.org/10.5194/hess-21-2649-2017

        Parameters:
        ===========
        obs_data : ArrayLike
            Observations data (the "truth" distribution to correct towards)
        mod_data : ArrayLike
            Model data (used in calculating the correction to adjust model data
            towards obs data distribution)
        distribution : Literal["normal", "gamma"]
            Type of distribution? (default normal)
        cdf_threshold : float
            Threshold for distribution tails (avoids infinity values)
        detrend_type : str
            Perform detrending ("linear") or constant offset ("constant")?

        Notes:
        ======
        Modified from bias correction pypy module:
        https://github.com/pankajkarman/bias_correction

        scaling factor - see equation 8 Switanek et al. 2017
        ri, ri_scaled (reccurence interval) - see equation 9 Switanek et al. 2017
        cdf_scaled - see equation 10 Switanek et al. 2017

        detrends - applies detrending to data (detrend_type constant or linear)
        cdf, cdf_detrends - CDF of raw data / detrended data
        inv_cdf, inv_cdf_detrends - inverse CDF of raw data / detrended data

        Usage:
        ======
        >>> adjustment = SDMBiasAdjust(obs, rcm_hist)
        >>> rcm_hist_corrected = adjustment.correct()
        >>> rcm_futr_corrected = adjustment.correct(rcm_futr)
        """
        super().__init__(obs_data, mod_data)
        self.distribution = distribution
        self.cdf_threshold = cdf_threshold
        self.detrend_type = detrend_type

        # initialise empty variables
        self.scaling_factor = None
        self.ri = None
        self.ri_scaled = None
        self.cdf_scaled = None

        # pre-processing
        self.detrends = self.data.apply(scipy.signal.detrend, type=detrend_type)

        # calculate the cdf, inverse cdf of modelled data
        self.cdf = pd.Series(index=self.data.index, dtype=object)
        self.cdf_detrends = pd.Series(index=self.data.index, dtype=object)
        self.inv_cdf = pd.Series(index=self.data.index, dtype=object)
        self.inv_cdf_detrends = pd.Series(index=self.data.index, dtype=object)
        for key in self.data.index:
            self.cdf[key] = fit_cdf(distribution, self.data[key], self.stats.loc[key])
            self.cdf_detrends[key] = fit_cdf(distribution, self.detrends[key])
            self.inv_cdf[key] = fit_inv_cdf(distribution, self.data[key], cdf=self.cdf[key])
            self.inv_cdf_detrends[key] = fit_inv_cdf(distribution, self.detrends[key], cdf=self.cdf_detrends[key])

    def normal_adjust(self) -> ArrayLike:
        # collect the input dataset alongside obs, mod
        data = self.data.copy()
        data["input"] = self.input

        # get length, mean, of original timeseries
        lens = data.apply(len)
        means = self.stats["mean"].copy()
        means["input"] = np.nanmean(self.input)

        # get detrended timeseries and normal distribution params (mean, std)
        detrends = self.detrends.copy()
        detrends["input"] = scipy.signal.detrend(self.input, type=self.detrend_type)
        dstats = calculate_statistics(detrends, ["mean", "std"])

        # CDF of each array
        cdf_detrends = self.cdf_detrends.copy()
        cdf_dtr = pd.Series(index=data.index, dtype=object)
        for key in data.index:
            norm_cdf = cdf_detrends.get(key)
            if norm_cdf is None:
                norm_cdf = fit_cdf("normal", stats=dstats.loc[key])
            cdf_dtr[key] = threshold_cdf(norm_cdf(np.sort(detrends[key])), self.cdf_threshold)

        # get target (model future period) diff and indexes, to apply subsequently after processing
        argsort_input = np.argsort(detrends["input"])
        diff_input = self.input - detrends["input"]

        # interpolation required based on timeperiod differences
        # note that input (mod future) will be unchanged
        cdf_interps = pd.Series(dtype=object)
        for key, arr_len in lens.items():
            cdf_interps[key] = np.interp(
                np.linspace(1, arr_len, lens["input"]),
                np.linspace(1, arr_len, arr_len),
                cdf_dtr[key]
            )

        # Eq. 9, 10
        # calculation steps for inverse, reccurance interval scaled CDFs
        # 0.5 shifted, inverse, and recurrence interval scaled CDFs
        self.ri = 1/(0.5 - np.abs(cdf_interps - 0.5))
        self.ri_scaled = self.ri["obs"] * self.ri["input"] / self.ri["mod"]
        self.ri_scaled[self.ri_scaled < 1] = 1
        adapted_cdf = 0.5 + np.sign(cdf_interps["obs"] - 0.5) * (0.5 - 1 / self.ri_scaled)
        self.cdf_scaled = np.sort(threshold_cdf(adapted_cdf, self.cdf_threshold))

        # Eq. 8
        # bias corrected values (unordered) 
        var_scaling = dstats["std"]["obs"] / dstats["std"]["mod"]
        adapted_inv_obs = fit_inv_cdf("normal", stats=dstats.loc["obs"])(self.cdf_scaled)
        input_inv_self = fit_inv_cdf("normal", stats=dstats.loc["input"])(cdf_dtr["input"])
        input_inv_mod = fit_inv_cdf("normal", stats=dstats.loc["mod"])(cdf_dtr["input"])
        self.scaling_factor = var_scaling * (input_inv_self - input_inv_mod)
        # Eq. 11
        xvals = adapted_inv_obs + self.scaling_factor
        xvals += means["obs"] - np.mean(xvals) + means["input"] - means["mod"]

        # correct order and insert trend/offset array
        correction = np.zeros(lens["input"])
        correction[argsort_input] = xvals
        correction += diff_input - means["input"]

        # return bias corrected array (scaled normal distribution mapping)
        # preserve original format (pandas or xarray)
        out = self.input * 0
        return out + correction

    def gamma_adjust(self) -> ArrayLike:
        raise NotImplementedError("Gamma adjust not implemented!")

    def correct(self, input: Optional[ArrayLike]=None) -> ArrayLike:
        # select appropriate method
        bias_adjustments = {
            "normal": self.normal_adjust,
            "gamma": self.gamma_adjust
        }
        # apply correction
        return super()._correct(bias_adjustments[self.distribution], input)

