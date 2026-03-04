import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import find_peaks
from pmt_analysis.processing.basics import FullWindow
import collections
from typing import Optional


class AfterPulses:
    """Afterpulse finding and processing.

    Attributes:
        input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).
        input_data_std: Standard deviation of the input data baseline per waveform.
        verbose: Set verbose output.
        adc_f: ADC sampling frequency in samples per second as provided by same attribute of
            `pmt_analysis.utils.input.ADCRawData`.
        occupancy: Occupancy of the main pulse in PE per (LED) trigger.
        occupancy_unc: Occupancy uncertainty of the main pulse in PE per (LED) trigger.
        area_thr_ap: Lower area threshold for afterpulse candidates.
        t_thr_ap: Lower time threshold for afterpulse candidates.
        n_samples: Total number of waveforms analyzed.
        df: Pandas dataframe with after pulse candidates and their properties.
        ap_rate_dict: Dictionary with resulting afterpulse rate value and uncertainty.
    """

    def __init__(self, input_data: np.ndarray, adc_f: float, verbose: bool = True,
                 pre_filter_threshold: float = 3, pre_filter_threshold_type: str = 'std',
                 occupancy: Optional[float] = None, occupancy_unc: Optional[float] = None,
                 area_thr_ap: Optional[float] = None, t_thr_ap: Optional[float] = None):
        """Init of the AfterPulses class.

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform) as
                provided by `pmt_analysis.utils.input.ADCRawData`.
            verbose: Set verbose output.
            pre_filter_threshold: Amplitude threshold to exclude waveforms that do not contain any entries above
                threshold (i.e. no main pulse or afterpulse of a certain minimum size) to make further processing
                more efficient. Set to a negative value to effectively disable pre-filtering.
            pre_filter_threshold_type: 'abs' for absolute threshold or 'std' for thresholds of multiples of the
                baseline standard deviation.
            adc_f: ADC sampling frequency in samples per second as provided by same attribute of
                `pmt_analysis.utils.input.ADCRawData`.
            occupancy: Occupancy of the main pulse in PE per (LED) trigger.
            occupancy_unc: Occupancy uncertainty of the main pulse in PE per (LED) trigger.
            area_thr_ap: Lower area threshold for afterpulse candidates.
            t_thr_ap: Lower time threshold for afterpulse candidates.
        """
        self.adc_f = adc_f
        self.occupancy = occupancy
        self.occupancy_unc = occupancy_unc
        self.area_thr_ap = area_thr_ap
        self.t_thr_ap = t_thr_ap
        self.input_data = input_data
        self.input_data_std = FullWindow().get_baseline_std(self.input_data)
        self.verbose = verbose
        self.n_samples = self.input_data.shape[0]
        self.df = pd.DataFrame()
        self.ap_rate_dict = {}
        # Remove waveforms containing no entries above some threshold (i.e. no main pulse or afterpulse).
        if pre_filter_threshold_type == 'std':
            self.pre_filter(amplitude_threshold_abs=None, amplitude_threshold_std=pre_filter_threshold)
        elif pre_filter_threshold_type == 'abs':
            self.pre_filter(amplitude_threshold_abs=pre_filter_threshold, amplitude_threshold_std=None)
        else:
            raise ValueError('Parameter pre_filter_threshold_type can only take values abs or std.')

    def pre_filter(self, amplitude_threshold_abs: Optional[float] = None,
                   amplitude_threshold_std: Optional[float] = 3):
        """Method to exclude waveforms that do not contain any entries above an amplitude threshold,
        i.e. no main pulse or afterpulse of a certain minimum size, in order to make further processing
        more efficient.

        Args:
            amplitude_threshold_abs: Absolute amplitude threshold. Default: None (disabled).
            amplitude_threshold_std: Amplitude threshold in units of baseline standard deviations.
                Ensure that the signal-to-noise ratio is sufficient for the selected value. Default: 3.
        """
        if (amplitude_threshold_abs is not None) and (amplitude_threshold_std is not None):
            raise ValueError('Either amplitude_threshold_abs or amplitude_threshold_std must be None '
                             'to be disabled, using both options is not permitted.')
        elif (amplitude_threshold_abs is None) and (amplitude_threshold_std is None):
            raise ValueError('Either amplitude_threshold_abs or amplitude_threshold_std must have '
                             'a finite value.')
        amplitudes = FullWindow().get_amplitude(self.input_data)
        if amplitude_threshold_abs is not None:
            if self.verbose:
                print('Pre-filtering data with absolute threshold {}.'.format(amplitude_threshold_abs))
            self.input_data = self.input_data[amplitudes > amplitude_threshold_abs]
        elif amplitude_threshold_std is not None:
            if self.verbose:
                print('Pre-filtering data with threshold at {} baseline '
                      'standard deviations.'.format(amplitude_threshold_std))
            thresholds = amplitude_threshold_std * self.input_data_std
            self.input_data = self.input_data[amplitudes > thresholds]
            self.input_data_std = self.input_data_std[amplitudes > thresholds]

    @staticmethod
    def convert_to_amplitude(input_data: np.ndarray) -> np.ndarray:
        """Convert input_data ADC values to baseline-subtracted and sign reversed values
        (i.e. equivalent to amplitude definition).

        Args:
            input_data: Array with ADC data of shape (number of waveforms, time bins per waveform).

        Returns:
            output: Baseline-subtracted and sign reversed input values.
        """
        baselines = FullWindow().get_baseline(input_data)
        output = - np.subtract(input_data.T, baselines).T

        return output

    def find_ap(self, height: float, distance: float = 6, prominence_std: float = 8):
        """Find waveforms with at least two pulses using `scipy.signal.find_peaks`.
        The `df` attribute will be expanded by the following columns:

        - `idx`: running index to indicate waveforms with more than two found pulses
        - `input_data_converted`: baseline-subtracted and sign reversed afterpulse candidate waveform values
        - `input_data_std`: baseline standard deviation
        - `p0_position`: index position of first found pulse
        - `p1_position`: index position of afterpulse candidate

        For large datasets, the input_data attribute gets overwritten to save memory, as it won't be used
        anymore thereafter, and it is still implicitly available through `df.input_data_converted`.

        Args:
            height: Required height of peaks. Fixed threshold above noise, deduced from amplitude spectrum.
            distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
                Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
            prominence_std: Required prominence of peaks in units of baseline standard deviations.
        """
        if self.verbose:
            print('Finding waveforms with afterpulse candidates.')
        input_data_converted = self.convert_to_amplitude(self.input_data)
        prominence = prominence_std * FullWindow().get_baseline_std(self.input_data)

        # Overwrite self.input_data for large data sets to save memory, as it won't be used anymore hereafter,
        # and it is still implicitly available through `df.input_data_converted`.
        if self.input_data.shape[0] > 5e5:
            warnings.warn('Overwriting input_data attribute to save memory, '
                          'use df.input_data_converted attribute instead.')
            self.input_data = np.array(['Removed to save memory, use df.input_data_converted attribute instead.'])

        # make an empty numpy array to fill
        n = input_data_converted.shape[0] * 10
        dt = np.dtype([('idx', np.int32), ('input_data_converted', object), ('input_data_std', np.float64), ('p0_position', np.int32), ('p1_position', np.int32)])
        arr = np.zeros(n, dtype=dt)

        idx = 0
        arr_idx = 0
        for i, el in tqdm(enumerate(input_data_converted)):  # TODO: try to vectorize
            peak_positions, _ = find_peaks(el,
                                           height=height,
                                           prominence=prominence[i],
                                           distance=distance)
            if peak_positions.shape[0] > 1:
                p0_position = peak_positions[0]
                input_data_std = self.input_data_std[i]
                for p1_position in peak_positions[1:]:
                    arr[arr_idx]['idx'] = idx
                    arr[arr_idx]['input_data_converted'] = el.tolist()
                    arr[arr_idx]['input_data_std'] = input_data_std
                    arr[arr_idx]['p0_position'] = p0_position
                    arr[arr_idx]['p1_position'] = p1_position
                    arr_idx += 1
                    # df_add = pd.DataFrame({'idx': [idx],
                    #                        'input_data_converted': [el.tolist()],
                    #                        'input_data_std': input_data_std,
                    #                        'p0_position': [p0_position],
                    #                        'p1_position': [p1_position]})
                    # self.df = pd.concat([self.df, df_add])
                idx += 1
        # trim
        arr = arr[:arr_idx]
        # self.df.reset_index(drop=True, inplace=True)
        self.df = pd.DataFrame(arr)

    def constrain_main_peak(self, trim: bool = True):
        """Identify whether the first found pulse is a viable candidate for the main pulse
        (e.g. the LED induced signal) based on its timing compared to the other main pulse candidates.

        Args:
            trim: If `True`, remove events where the first found pulse is not a viable candidate for the main pulse and
                hence the afterpulse event may be misidentified. If `False` only add column `valid_main_pulse` to `df`.
        """
        # Define relevant percentiles
        lp = np.percentile(self.df['p0_position'], (100 - 68.27) / 2)
        cp = np.percentile(self.df['p0_position'], 50)
        up = np.percentile(self.df['p0_position'], 100 - (100 - 68.27) / 2)
        sf = 5  # acceptable Gaussian standard deviations from median
        lt = cp - sf * (cp - lp)
        ut = cp + sf * (up - cp)

        # Apply main peak thresholds
        valid_main_pulse = (self.df['p0_position'] >= lt) & (self.df['p0_position'] <= ut)
        if trim:
            if self.verbose:
                print('Constraining main peak.')
            self.df = self.df[valid_main_pulse]
            self.df.drop(['valid_main_pulse'], axis=1, errors='ignore', inplace=True)
        else:
            self.df['valid_main_pulse'] = valid_main_pulse

    def get_ap_properties(self):
        """Calculate properties of found main pulse and afterpulse candidates and add corresponding columns to `df`.

        - `t_diff_ns`: Temporal difference main pulse and afterpulse in ns.
        - `p0_amplitude`: Amplitude of the main pulse candidate (in ADC bins).
        - `p1_amplitude`: Amplitude of the afterpulse candidate (in ADC bins).
        - `p0_lower_bound`: Lower bound of main pulse candidate integration window.
        - `p0_upper_bound`: Upper bound of main pulse candidate integration window.
        - `p0_area`: Area of main pulse candidate.
        - `p1_lower_bound`: Lower bound of afterpulse candidate integration window.
        - `p1_upper_bound`: Upper bound of afterpulse candidate integration window.
        - `p1_area`: Area of afterpulse candidate.
        - `separable`: Boolean stating whether main pulse and afterpulse candidates are separable and not overlapping.
            Only if true, area and amplitude calculations are reliable.
        """
        if self.verbose:
            print('Calculating peak properties.')
        # Temporal difference main pulse and afterpulse
        self.df['t_diff_ns'] = (self.df['p1_position'] - self.df['p0_position']) / (self.adc_f * 1e-9)

        # Amplitudes found pulses
        self.df['p0_amplitude'] = np.array(self.df['input_data_converted'].tolist())[np.arange(self.df.shape[0]),
                                                                                     np.array(self.df['p0_position'])]
        self.df['p1_amplitude'] = np.array(self.df['input_data_converted'].tolist())[np.arange(self.df.shape[0]),
                                                                                     np.array(self.df['p1_position'])]
        # Peak ranges and areas
        valley = np.zeros(self.df.shape[0])
        p0_lower_bound = np.zeros(self.df.shape[0], dtype=int)
        p0_upper_bound = np.zeros(self.df.shape[0], dtype=int)
        p0_area = np.zeros(self.df.shape[0])
        p1_lower_bound = np.zeros(self.df.shape[0], dtype=int)
        p1_upper_bound = np.zeros(self.df.shape[0], dtype=int)
        p1_area = np.zeros(self.df.shape[0])
        separable = np.zeros(self.df.shape[0], dtype=bool)

        for i, el in enumerate(self.df['input_data_converted']):
            # Position of the lowest value between identified main pulse and afterpulse candidates
            valley[i] = np.argmin(el[self.df['p0_position'].iloc[i]:self.df['p1_position'].iloc[i]]) + \
                        self.df['p0_position'].iloc[i]

            # Lower window bound for main pulse candidate area estimation, take third last entry below
            # one sigma above baseline before main pulse candidate
            try:
                p0_lower_bound[i] = np.where(el[:self.df['p0_position'].iloc[i]] < self.input_data_std[i])[0][-3]
            except IndexError:
                p0_lower_bound[i] = 0  # unconstrained

            # Upper window bound for main pulse candidate area estimation, take third entry below
            # one sigma above baseline after main pulse candidate or 'valley' value
            try:
                p0_upper_bound[i] = (np.where(el[self.df['p0_position'].iloc[i]:] < self.input_data_std[i])[0][2]) + \
                                    self.df['p0_position'].iloc[i]
            except IndexError:
                p0_upper_bound[i] = valley[i]  # take 'valley' value
            p0_upper_bound[i] = min(p0_upper_bound[i], valley[i])

            # Lower window bound for afterpulse candidate area estimation, take third last entry below
            # one sigma above baseline before afterpulse candidate or 'valley' value
            try:
                p1_lower_bound[i] = np.where(el[:self.df['p1_position'].iloc[i]] < self.input_data_std[i])[0][-3]
            except IndexError:
                p1_lower_bound[i] = valley[i]  # take 'valley' value
            p1_lower_bound[i] = max(p1_lower_bound[i], valley[i])

            # Upper window bound for afterpulse candidate area estimation, take third entry below
            # one sigma above baseline after afterpulse candidate
            try:
                p1_upper_bound[i] = (np.where(el[self.df['p1_position'].iloc[i]:] < self.input_data_std[i])[0][2]) + \
                                    self.df['p1_position'].iloc[i]
            except IndexError:
                p1_upper_bound[i] = len(el) - 1  # unconstrained

            # Peak areas
            p0_area[i] = np.sum(el[p0_lower_bound[i]:p0_upper_bound[i] + 1])
            p1_area[i] = np.sum(el[p1_lower_bound[i]:p1_upper_bound[i] + 1])

            # Consider main pulse and afterpulse candidates separable if a minimum value between their respective
            # positions of at most 5% main pulse or afterpulse candidate height is reached.
            separable[i] = (el[int(valley[i])] <= 0.05*max(self.df.iloc[i]['p0_amplitude'],
                                                           self.df.iloc[i]['p1_amplitude']))

        self.df['p0_lower_bound'] = p0_lower_bound
        self.df['p0_upper_bound'] = p0_upper_bound
        self.df['p0_area'] = p0_area
        self.df['p1_lower_bound'] = p1_lower_bound
        self.df['p1_upper_bound'] = p1_upper_bound
        self.df['p1_area'] = p1_area
        self.df['separable'] = separable

    def multi_ap(self):
        """Find multi-afterpulse candidates that are merged in the area estimation.

        Split separable multi-afterpulse candidates in the same waveform and remove duplicate waveforms
        of non-separable, merged multi-afterpulse candidates to avoid double counting
        """
        idx_multi_ap = [item for item, count in collections.Counter(self.df.idx.tolist()).items() if count > 1]
        if self.verbose:
            print('Finding multi-afterpulse candidates that are merged in the area estimation.')
            print('Found total of {} multi-afterpulse candidate waveforms out of {} ({} distinct) general afterpulse '
                  'candidate waveforms.'.format(len(idx_multi_ap), self.df.shape[0], len(np.unique(self.df.idx))))
        if len(idx_multi_ap) > 0:
            n_rows = self.df.shape[0]
            # Entries to be modified, only alter values after all iterations
            mods = {'index': [], 'param_name': [], 'param_val': []}
            duplicates = []
            for i, el in enumerate(self.df['input_data_converted']):
                # Check if following afterpulse candidate in same waveform
                if (i < n_rows - 1) and (self.df['idx'].iloc[i] == self.df['idx'].iloc[i + 1]):
                    # Check if identified with same integration window
                    if (self.df['p1_lower_bound'].iloc[i] == self.df['p1_lower_bound'].iloc[i + 1]) and (
                            self.df['p1_upper_bound'].iloc[i] == self.df['p1_upper_bound'].iloc[i + 1]):
                        # Position of the lowest value between present and subsequent afterpulse candidate
                        valley = np.argmin(el[self.df['p1_position'].iloc[i]:self.df['p1_position'].iloc[i + 1]]) + \
                                 self.df['p1_position'].iloc[i]
                        # Check if present and subsequent afterpulse candidate separable,
                        # i.e. valley at most 30% of both peaks
                        separable = (el[valley] <= 0.3 * self.df['p1_amplitude'].iloc[i]) and (
                                    el[valley] <= 0.3 * self.df['p1_amplitude'].iloc[i + 1])
                        if separable:
                            mods['index'].append(self.df.index[i])
                            mods['param_name'].append('p1_upper_bound')
                            mods['param_val'].append(valley)
                # Check if previous afterpulse candidate in same waveform
                if (i > 0) and (self.df['idx'].iloc[i] == self.df['idx'].iloc[i - 1]):
                    # Check if identified with same integration window
                    if (self.df['p1_lower_bound'].iloc[i] == self.df['p1_lower_bound'].iloc[i - 1]) and (
                            self.df['p1_upper_bound'].iloc[i] == self.df['p1_upper_bound'].iloc[i - 1]):
                        # Position of the lowest value between present and subsequent afterpulse candidate
                        valley = np.argmin(el[self.df['p1_position'].iloc[i - 1]:self.df['p1_position'].iloc[i]]) + \
                                 self.df['p1_position'].iloc[i - 1]
                        # Check if present and subsequent afterpulse candidate separable,
                        # i.e. valley at most 30% of both peaks
                        separable = (el[valley] <= 0.3 * self.df['p1_amplitude'].iloc[i - 1]) and (
                                    el[valley] <= 0.3 * self.df['p1_amplitude'].iloc[i])
                        if separable:
                            mods['index'].append(self.df.index[i])
                            mods['param_name'].append('p1_lower_bound')
                            mods['param_val'].append(valley)
                        else:
                            duplicates.append(self.df.index[i])

            # Modify integration bounds for separable multi-afterpulse candidates and update area estimate accordingly
            if len(mods['index']) > 0:
                for i, ind in enumerate(mods['index']):
                    self.df.at[ind, mods['param_name'][i]] = mods['param_val'][i]
                    self.df.at[ind, 'p1_area'] = np.sum(
                        (self.df.at[ind, 'input_data_converted'])[self.df.at[ind, 'p1_lower_bound']:
                                                                  self.df.at[ind, 'p1_upper_bound'] + 1])

            # Remove duplicate waveforms of non-separable, merged multi-afterpulse candidates to avoid double counting
            self.df.drop(index=duplicates, inplace=True)

            if self.verbose:
                print(
                    'Found {} yet unsplit afterpulse candidates and removed {} duplicate waveforms of merged '
                    'multi-afterpulse candidates.'.format(len(mods['index']), len(duplicates)))
        else:
            if self.verbose:
                print('No merged multi-afterpulse candidate waveforms found.')

    def ap_rate(self):
        """Calculate afterpulse rates, normalized to the occupancy.

        - `n_ap`: Total number of identified afterpulses.
        - `n_ap_separable`: Total number of identified afterpulses separable from the main pulse.
        - `ap_fraction`: Fraction of waveforms with identified afterpulse(s).
        - `ap_fraction_unc`: Statistical uncertainty of the fraction of waveforms with identified afterpulse(s).
        - `ap_fraction_separable`: Fraction of waveforms with identified afterpulses separable from the main pulse.
        - `ap_fraction_separable_unc`: Statistical uncertainty of the fraction of waveforms with identified \
          afterpulses separable from the main pulse.
        - `ap_rate`: Afterpulse probability in units of afterpulses per PE.
        - `ap_rate_unc`: Uncertainty of the afterpulse probability in units of afterpulses per PE.
        - `ap_rate_separable`: Afterpulse probability in units of afterpulses (separable from the main pulse) per PE.
        - `ap_rate_separable_unc`: Uncertainty of the afterpulse probability in units of afterpulses \
          (separable from the main pulse) per PE.
        - `area_thr_ap`: Lower area threshold for afterpulse candidates.
        - `t_thr_ap`: Lower time threshold for afterpulse candidates.
        - `ap_rate_separable_above_thr`: Afterpulse probability in units of afterpulses (separable from the main pulse)
          per PE for afterpulse candidates with areas above `area_thr_ap` and time differences above `t_thr_ap`.
        - `ap_rate_separable_unc_above_thr`: Uncertainty of the afterpulse probability in units of afterpulses
          (separable from the main pulse) per PE for afterpulse candidates with areas above `area_thr_ap`
          and time differences above `t_thr_ap`.
        """
        n_ap = self.df.shape[0]
        n_ap_separable = np.sum(self.df.separable)
        ap_fraction = n_ap / self.n_samples
        ap_fraction_unc = np.sqrt(n_ap) / self.n_samples
        ap_fraction_separable = n_ap_separable / self.n_samples
        ap_fraction_separable_unc = np.sqrt(n_ap_separable) / self.n_samples
        if self.occupancy is None:
            warnings.warn('No occupancy given, afterpulse rate will not be fully calculated.')
            ap_rate = None
            ap_rate_unc = None
            ap_rate_separable = None
            ap_rate_separable_unc = None
            ap_rate_separable_above_thr = None
            ap_rate_separable_unc_above_thr = None
        else:
            ap_rate = ap_fraction / self.occupancy
            ap_rate_separable = ap_fraction_separable / self.occupancy
            if self.occupancy_unc is None:
                warnings.warn('No occupancy uncertainty given, will assume negligible occupancy uncertainty '
                              'for afterpulse rate calculation.')
                ap_rate_unc = ap_fraction_unc / self.occupancy
                ap_rate_separable_unc = ap_fraction_separable_unc / self.occupancy
            else:
                ap_rate_unc = ap_rate * np.sqrt((ap_fraction_unc/ap_fraction)**2
                                                + (self.occupancy_unc/self.occupancy)**2)
                ap_rate_separable_unc = ap_rate_separable * np.sqrt((ap_fraction_separable_unc /
                                                                     ap_fraction_separable) ** 2
                                                                    + (self.occupancy_unc/self.occupancy) ** 2)
            if self.area_thr_ap is None:
                warnings.warn('No lower area threshold for afterpulse candidates provided.')
                ap_rate_separable_above_thr = None
                ap_rate_separable_unc_above_thr = None
            else:
                if self.t_thr_ap is None:
                    warnings.warn('No lower time threshold for afterpulse candidates provided, '
                                  'will not apply temporal selection.')
                    n_ap_separable_above_thr = np.sum(self.df.separable & (self.df.p1_area >= self.area_thr_ap))
                else:
                    n_ap_separable_above_thr = np.sum(self.df.separable
                                                      & (self.df.p1_area >= self.area_thr_ap)
                                                      & (self.df.t_diff_ns >= self.t_thr_ap))
                ap_fraction_separable_above_thr = n_ap_separable_above_thr / self.n_samples
                ap_fraction_separable_unc_above_thr = np.sqrt(n_ap_separable_above_thr) / self.n_samples
                ap_rate_separable_above_thr = ap_fraction_separable_above_thr / self.occupancy
                if self.occupancy_unc is None:
                    ap_rate_separable_unc_above_thr = None
                else:
                    ap_rate_separable_unc_above_thr = (ap_rate_separable_above_thr
                                                       * np.sqrt((ap_fraction_separable_unc_above_thr /
                                                                  ap_fraction_separable_above_thr) ** 2
                                                                 + (self.occupancy_unc / self.occupancy) ** 2))

        self.ap_rate_dict = {'n_ap': n_ap, 'n_ap_separable': n_ap_separable,
                             'ap_fraction': ap_fraction, 'ap_fraction_unc': ap_fraction_unc,
                             'ap_fraction_separable': ap_fraction_separable,
                             'ap_fraction_separable_unc': ap_fraction_separable_unc,
                             'ap_rate': ap_rate, 'ap_rate_unc': ap_rate_unc,
                             'ap_rate_separable': ap_rate_separable, 'ap_rate_separable_unc': ap_rate_separable_unc,
                             'area_thr_ap': self.area_thr_ap, 't_thr_ap': self.t_thr_ap,
                             'ap_rate_separable_above_thr': ap_rate_separable_above_thr,
                             'ap_rate_separable_unc_above_thr': ap_rate_separable_unc_above_thr
                             }

    def compute(self, height: float, distance: float = 6, prominence_std: float = 8,
                constrain_main_peak: bool = True):
        """Perform full afterpulse analysis.

        Args:
            height: Required height of peaks. Fixed threshold above noise, deduced from amplitude spectrum.
            distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks.
                Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.
            prominence_std: Required prominence of peaks in units of baseline standard deviations.
            constrain_main_peak: Remove events where the first found pulse is not a viable candidate for
                the main pulse.
        """
        self.find_ap(height=height, distance=distance, prominence_std=prominence_std)
        self.constrain_main_peak(trim=constrain_main_peak)
        self.get_ap_properties()
        self.multi_ap()
        self.ap_rate()
