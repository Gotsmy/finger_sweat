# Finger Sweat Analysis Enables Short Interval Metabolic Biomonitoring in Humans 
This is code discussed in Brunmair et al. 2020 [1]. For further explaination please see the manuscript and supplementary notes "Mathematical Model" and "Sensitivity Analysis".

## Prerequisites
Python 3.7 and packages listed in requirements.txt.
```
pip install -r requirements.txt
```

## Mathematical Model
Here we fitted our model onto raw time-series data.

### Example
The notebook `mathematical_model/donor_fitting.ipynb` is used to fit kinetic/ concentration/ V<sub>sweat</sub> parameters of our model onto the measured (IS normalized) data (M&#771;) given in `mathematical_model/raw_data/`.
Custom python functions are listed in the notebook in section `functions`. In `sampling for MC replicates & multiprocess fitting` the actual fitting is done and the results are saved in the `mathematical_model/fitted_parameters/` directory. In sections `parse results` and `plot results` best solution for every time-series is read in and the unit-less concentrations of caffeine, paraxanthine, theobromine, and theophylline (as described in the Supplementary Notes: Mathematical Model) are plotted.

### Data Availability
The notebook `mathematical_model/donor_analysis.ipynb` provides code that has been used to create figures shown in the manuscript and supplementary material. There we used donor 1 and donor 2 as examples, here the figures are plotted for all donors.


## Sensitivity Analysis
Here we analysed the error associated to the fitting procedure with theoretical simulations.

### Example
To run a test simulation go to `sensitivity_analysis/test` and run the `run_simulation.py` script. Note: with standard settings this can take several hours, depending on your machine. To run a shorter test consider lowering the values for *n* and *i* (`max_reps`and `max_tries` respectively).
```
cd sensitivity_analysis/test
python ../run_simulation.py
```
This will create a new run_simulation.txt file with information about original kinetic & concentration parameters/ timepoints/ V<sub>sweat</sub> and error values as well as the best solution parameters per bootstrap replicate. Optionally, it will create a `run_simulation_raw` directory with *i* files that store all *n*·(11+*j*) fitting parameters and *n* losses.

### Data Availability
The table `run_information.csv` gives an overview of the runs simulated for the data shown in the manuscript and their settings and names. The raw data of the runs is given in `sensitivity_analysis/runs_manuscript/`. The notebook `sensitivity_analysis/sensitivity_analysis.ipynb` shows how the data was processed to extract CV and MRE of the different simulations. The resulting tables are found in the `sensitivity_analysis/analysis_tables/` directory. They correspond to the tables shown in the Supplementary Notes: Sensitivity Analysis section of the manuscript.

## License
The code (and especially the implemented loss functions) heavily relies on the `scipy`[2] and `robust_loss_pytorch`[3] packages with corresponding licenses found [here](https://www.scipy.org/scipylib/license.html) and  [here](https://github.com/jonbarron/robust_loss_pytorch/blob/master/LICENSE) repsectively. Original code is licensed under GPL v3 (see LICENSE file).

## References

[1] Brunmair, J., Niederstaetter, L., Neuditschko, B., Bileck, A., Slany, A., Janker, L., ... & Gerner, C. (2020). Finger Sweat Analysis Enables Short Interval Metabolic Biomonitoring in Humans. bioRxiv. https://doi.org/10.1101/2020.11.06.369355

[2] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.

[3] Barron, J. T. (2019). A general and adaptive robust loss function. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4331-4339).
