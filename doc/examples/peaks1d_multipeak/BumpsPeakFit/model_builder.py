"""
Author: A. J. Caruana (STFC, ISIS)

Model builder class for 1d multi peak fitting.
"""

import pandas as pd
import numpy as np
from .peaks_1d import Voigt, Gaussian, Background, Peaks


class ModelBuilder1d:

    def __init__(self, data_filename, peaks_filename):

        self.peak_table = self.peak_loader(peaks_filename)
        self.x, self.y, self.e = self.data_loader(data_filename)
        self.model = self.build_model()
        self.amps, self.xcs, self.sigs, self.gams = self._unpack_peak_values()
        self.set_values()

    @staticmethod
    def data_loader(filename):
        """
        Loads and returns XYE data as np.array
        Assumes basic data format:
        X, Y, E and no header
        """
        return np.loadtxt(filename).T

    @staticmethod
    def peak_loader(filename):
        """
        Loads in a .csv of peak positions using pandas

        This currently is fragile and will only work with a csv file 
        formatted as follows: 
        Header: 'PeakName','PeakType',  'Amplitude','Centre', 'Sigma', 'Gamma'
        Data:    'str'    ,'class type', 'float'   ,'float' , 'float', 'float'
        """
        # TODO: maybe select on order rather than name - fields=[0:4] and manually specify the header as before?
        fields = ["PeakName", "PeakType", "Amplitude", "Centre", "Sigma", "Gamma"]

        return pd.read_csv(f"{filename}", usecols=fields)

    def build_model(self):
        """
        Generates a Peaks() fitting instance from loaded .csv
        """

        peaks_list = self._get_peaks()

        return Peaks(parts=peaks_list,
                     X=self.x, data=self.y, err=self.e, plot_peaks=True)

    def set_values(self):
        """
        Set values from peak spec table 
        """

        for i, (peak, xc, A, sig, gam) in enumerate(zip(self.model.parts, self.xcs, self.amps, self.sigs, self.gams)):
            if isinstance(peak, Voigt):
                values = xc, A, sig, gam
                self._set_values_voigt(values, peak)
            elif isinstance(peak, Background):
                values = A
                self._set_values_background(values, peak)
            elif isinstance(peak, Gaussian):
                values = xc, A, sig
                self._set_values_gaussian(values, peak)

    def _unpack_peak_values(self):
        """
        Unpacks the numerical columns of the peak table and converts them to float
        If a non numeric value is found (i.e. can't convert to float) then is set to nan.
        """
        return [pd.to_numeric(self.peak_table[column], errors='coerce')
                for column in self.peak_table[["Amplitude", "Centre", "Sigma", "Gamma"]]]

    def _get_peaks(self):
        peaks = [self.peak_init(peak_type, peakname)
                 for peakname, peak_type in zip(self.peak_table["PeakName"], self.peak_table["PeakType"])]

        return peaks

    @staticmethod
    def _set_values_voigt(values, voigt_instance):
        # TODO: can we use inspect somehow to return the attributes 
        #  so that we an cycle through them in a more generic way?

        xc, A, sig, gam = values

        voigt_instance.xc.value = xc
        voigt_instance.A.value = A
        voigt_instance.sig.value = sig
        voigt_instance.gam.value = gam

    @staticmethod
    def _set_values_gaussian(values, gaussian_instance):
        # TODO: can we use inspect somehow to return the attributes 
        #  so that we an cycle through them in a more generic way?

        xc, A, sig = values
        gaussian_instance.xc.value = xc
        gaussian_instance.A.value = A
        gaussian_instance.sig.value = sig

    @staticmethod
    def _set_values_background(values, background_instance):

        A = values
        background_instance.C.value = A

    @staticmethod
    def peak_init(peakstr, peakname):
        """
        Takes keyword string and instantiates peak class
        """
        # TODO: could interrogate the peaks1d module for Peak types and iterate through?
        name = f"{peakname}"
        if peakstr == "Gaussian":
            return Gaussian(name=name)
        elif peakstr == "Voigt":
            return Voigt(name=name)
        elif peakstr == "Background":
            return Background(name=name)
        else:
            raise ValueError("Peak not recognised. Enter: Gaussian, Voigt or Background")
