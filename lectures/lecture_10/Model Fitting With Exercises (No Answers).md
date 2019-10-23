

```python
%matplotlib inline
import tellurium as te
import numpy as np
import lmfit   # Fitting lib
import math
import random 
import matplotlib.pyplot as plt
import model_fitting as mf
```


```python
# Model
model = """
     A -> B; k1*A
     B -> C; k2*B
      
     A = 5;
     B = 0;
     C = 0;
     k1 = 0.1
     k2 = 0.2
"""
parameters = mf.makeParameters(constants=['k1', 'k2'])
```


```python
# Create synthetic observational data and plot it.
true_data = mf.runSimulation(model=model, num_points=10)
obs_data = mf.makeObservations(model=model, noise_std=0.5, num_points=10)
columns = ['A', 'B', 'C']
mf.plotTimeSeries(true_data, title="True Model", columns=columns)
mf.plotTimeSeries(obs_data, title="Model With Noise", is_scatter=True, columns=columns)
```


![png](Model%20Fitting%20With%20Exercises%20%28No%20Answers%29_files/Model%20Fitting%20With%20Exercises%20%28No%20Answers%29_2_0.png)



![png](Model%20Fitting%20With%20Exercises%20%28No%20Answers%29_files/Model%20Fitting%20With%20Exercises%20%28No%20Answers%29_2_1.png)



```python
# Parameter fitting
# Illustration of parameter fitting
mf.fit(obs_data, model=model, parameters=parameters)
```




<table><tr><th> name </th><th> value </th><th> standard error </th><th> relative error </th><th> initial value </th><th> min </th><th> max </th><th> vary </th></tr><tr><td> k1 </td><td>  0.10270561 </td><td>  0.00899815 </td><td> (8.76%) </td><td> 1 </td><td>  0.00000000 </td><td>  10.0000000 </td><td> True </td></tr><tr><td> k2 </td><td>  0.21648761 </td><td>  0.03660162 </td><td> (16.91%) </td><td> 1 </td><td>  0.00000000 </td><td>  10.0000000 </td><td> True </td></tr></table>




```python
# Cross validate to fit model
mf.crossValidate(obs_data, model=model, parameters=parameters, num_folds=3)
```




    ([Parameters([('k1',
                   <Parameter 'k1', value=0.10473728834665352 +/- 0.00786, bounds=[0:10]>),
                  ('k2',
                   <Parameter 'k2', value=0.1811682768111761 +/- 0.0231, bounds=[0:10]>)]),
      Parameters([('k1',
                   <Parameter 'k1', value=0.10025984094188034 +/- 0.0086, bounds=[0:10]>),
                  ('k2',
                   <Parameter 'k2', value=0.1875837725727919 +/- 0.0282, bounds=[0:10]>)]),
      Parameters([('k1',
                   <Parameter 'k1', value=0.09792532982344115 +/- 0.00729, bounds=[0:10]>),
                  ('k2',
                   <Parameter 'k2', value=0.1966459415537608 +/- 0.027, bounds=[0:10]>)])],
     [0.9550082257729575, 0.9727385902021672, 0.9028996804274075])



## Exercise 1: Effect of Observational Data
1. Re-run the foregoing with nose_std=2.0. How do the fits change? How do the $R^2$ values change? Run the codes a few times to see the variations in the quality of fit and parameter estimates.
1. Do you get better fits if you increase the number of points?

## Exercise 2: Analyze a different model

     A -> B; k1*A
     B -> C; k2*B
     A -> C; k3*C
      
     A = 5;
     B = 0;
     C = 0;
     k1 = 0.1
     k2 = 0.2
     k3 = 0.3
     
 1. Create synthetic data for this model.
 1. How do the dynamics of the second model differ from the first?
 1. Do cross validation using this model and obtain $R^2$ values and parameter estimates for 2 folds? 3 folds?

## Exercise 3: Fitting the Wrong Model
1. Create synthetic data using the second model.
1. Fit the first model to these data.
1. How do the $R^2$ values from cross validation compare with those when we use the correct model? How accurately are k1 and k2 estimated?
1. What happens to parameter estimates if k3 = 0.1?
