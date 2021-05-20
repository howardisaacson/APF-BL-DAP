# APF-BL-DAP
### Automated Planet Finder - Breakthrough Listen - Data Analysis Pipeline

The purpose of this pipeline is to consolidate the various analysis packages that we use in Breakthrough Listen to analyze our high resolution spectra. Here are some of those steps

* Using the 1D, reduced spectra, we will run [SpecMatch-Emp](https://github.com/samuelyeewl/specmatch-emp]) on every spectrum to obtain stellar properties and produce spectra that are blaze corrected and registered to the observatory rest frame.

* Searching for optical laser emission from beyond the Earth is the primary science objective for Breakthrough Listen on the APF. This repo will hold laser detection and also injection/recovery routines.

* Spectral emisssion features from the Earth's atmosphere are problematic in the search for laser emission features as they very nearly match the spatial profile of the star. By properly modeling atmosphereic absorption features, we can more sensitively search for artificial laser emission.
