"""Radar vital-sign processing for one and several people (TI IWR6843AOP).

Modules
    sivis       SiViS file names and raw .bin reading
    cube        chirp settings, FFTs, static removal, virtual array
    spatial     angle spectra, range-angle maps, target picking, clustering, SBL
    beams       delay-and-sum, MVDR and LCMV beams, fixed beds, range sectors
    vitals      phase demodulation, filters and rate finders
    separation  ICA across beams, empirical wavelet transform
    assign      matching estimates to people
    learn       features and the learning models of Methods C and E
    simulate    scene simulator used for the figures
"""

__version__ = "1.0"
