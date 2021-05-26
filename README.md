# Finger Sweat Analysis Enables Short Interval Metabolic Biomonitoring in Humans 
This is code discussed in Brunmair et al. 2020 [1]. For further explaination please see the manuscript and supplementary notes "Mathematical Model" and "Sensitivity Analysis".

## Prerequisites
Python 3.7 and packages listed in requirements.txt.
```
pip install -r requirements.txt
```

## Mathematical Model
### Example & Data Availability
To fit kinetic/ concentration/ V<sub>sweat</sub> parameters of our model onto the measured data (M&#771;) given in `mathematical_model/raw_data/` use the notebook `mathematical_model/donor_fitting.ipynb`.
The custom python functions used are listed in the section `functions`. In `sampling for MC replicates & multiprocess fitting` the actual fitting is done and the results are saved in the `mathematical_model/fitted_parameters/` directory. In sections `parse results` and `plot results` the best solution for every time-series is read in and plotted.

## Supplementary Note: Sensitivity Analysis
Here we analysed the error associated to the fitting procedure with theoretical simulations.

### Example
To run a test simulation go to `sensitivity_analysis/test` and run the `run_simulation.py` script.
```
cd sensitivity_analysis/test
python ../run_simulation.py
```
This will create a new run_simulation.txt file with information about original kinetic & concentration parameters/ timepoints/ V<sub>sweat</sub> and error values as well as the best solution parameters per bootstrap replicate. Optionally, it will create a `run_simulation_raw` directory with *i* files that store all *n*·(11+*j*) fitting parameters and *n* losses.

### Data Availability
The table `run_information.csv` gives an overview of the runs simulated for the data shown in the manuscript and their settings and names. The raw data of the runs is given in `/sensitivity_analysis/runs_manuscript/`.

## License
The code heavily relies on the `scipy`[2] and `robust_loss_pytorch`[3] packages with corresponding licenses found [here](https://www.scipy.org/scipylib/license.html) and  [here](https://github.com/jonbarron/robust_loss_pytorch/blob/master/LICENSE) repsectively. Original code is licensed under GPL v3 (see LICENSE file).

## References

[1] Brunmair, J., Niederstaetter, L., Neuditschko, B., Bileck, A., Slany, A., Janker, L., ... & Gerner, C. (2020). Finger Sweat Analysis Enables Short Interval Metabolic Biomonitoring in Humans. bioRxiv. https://doi.org/10.1101/2020.11.06.369355

[2] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.

[3] Barron, J. T. (2019). A general and adaptive robust loss function. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4331-4339).
