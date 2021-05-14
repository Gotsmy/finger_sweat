# Finger Sweat Analysis Enables Short Interval Metabolic Biomonitoring in Humans 

## Prerequisites
Python 3.7 and packages listed in requirements.txt.
```
pip install -r requirements.txt
```

## Mathematical Model
### Example
To fit kinetic/ concentration/ V<sub>sweat</sub> parameters of our model onto the measured data (M&#771;) given in `mathematical_model/raw_data/` use the notebook `mathematical_model/donor_fitting.ipynb`.
The custom python functions used are listed in the section `functions`. In `sampling for MC replicates & multiprocess fitting` the actual fitting is done and the results are saved in the `mathematical_model/fitted_parameters/` directory. In sections `parse results` and `plot results` the best solution for every time-series is read in and plotted.

### Data Availability
Since the Monte Carlo replicates are randomly sampled small differences of the solutions can occur between two runs. Therefore, the results which were used for the manuscript figures are given in `mathematical_model/fitted_parameters_manuscript/`.


## Supplementary Note: Sensitivity Analysis
Here we analysed the error associated to the fitting procedure with theoretical simulations.

### Example
To test the script go to `sensitivity_analysis/test` and run the `run_simulation.py` script.
```
cd sensitivity_analysis/test
python ../run_simulation.py
```
This will create a new run_simulation.txt file with information about original kinetic & concentration parameters/ timepoints/ V<sub>sweat</sub> and error values as well as the best solution parameters per bootstrap replicate. Optionally, it will create a `run_simulation_raw` directory with *i* files that store all *n* fitting parameters and losses.

### Data Availability
The table `run_information.csv` gives an overview of the runs simulated for the data shown in the manuscript and their settings and names. The raw data of the runs is given in `/sensitivity_analysis/runs_manuscript/`.

## License
The code heavily relies on the `scipy`[1] and `robust_loss_pytorch`[2] packages.

[1] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.

[2] Barron, J. T. (2019). A general and adaptive robust loss function. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4331-4339).

TODO