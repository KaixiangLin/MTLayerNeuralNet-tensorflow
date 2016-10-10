# MTLayerNeuralNet-tensorflow
Multi layer fully connected neural network in tensorflow for arbitrary number of hidden layers - a simple example.

## Prerequisites
```
Python 2
Tensorflow
```

## Usage:
1. Run </br>
Simply Type following command
    ```python
    python MTLayer_run.py 10  # specify using 10 hidden layers
    ```
    or
    ```python
    python MTLayer_run.py  # use default value 3 hidden layers
    ```

2. Tune parameters </br>
Change parameters settings in `MTLayer_configure.py`.  e.g. In Line 18th: choose 1 to use sigmoid activation function and 0 to use tanh activation function.

3. Results </br>
The results will be printed on terminal as follows and a objective function convergence curve will be ploted in folder named by current time in data folder.

```
 The current time is : 20161010_19-38 
 
Step 0: obj = 265.87 reg = 13.98 (0.018 sec)
Step 1: obj = 264.15 reg = 13.95 (0.003 sec)
Step 100: obj = 167.20 reg = 11.95 (0.004 sec)
Step 200: obj = 152.07 reg = 11.87 (0.006 sec)
Step 300: obj = 146.92 reg = 12.41 (0.003 sec)
Step 400: obj = 142.50 reg = 13.49 (0.003 sec)
Step 500: obj = 137.92 reg = 15.17 (0.004 sec)
Step 600: obj = 133.09 reg = 17.47 (0.003 sec)
Step 700: obj = 127.55 reg = 20.50 (0.003 sec)
Step 800: obj = 121.27 reg = 24.10 (0.003 sec)
Step 900: obj = 115.35 reg = 27.94 (0.003 sec)
Training Data Eval:
Step 999: training error  = 110.30 
Step 1000: obj = 110.28 reg = 31.81 (0.004 sec)
Step 1100: obj = 105.58 reg = 35.81 (0.003 sec)
Step 1200: obj = 101.14 reg = 40.02 (0.004 sec)
Step 1300: obj = 97.22 reg = 44.39 (0.003 sec)
Step 1400: obj = 93.80 reg = 48.74 (0.003 sec)
Step 1500: obj = 90.58 reg = 53.08 (0.007 sec)
Step 1600: obj = 87.44 reg = 57.57 (0.003 sec)
Step 1700: obj = 84.73 reg = 62.12 (0.003 sec)
Step 1800: obj = 82.57 reg = 66.47 (0.004 sec)
Step 1900: obj = 80.90 reg = 70.49 (0.003 sec)
Training Data Eval:
Step 1999: training error  = 79.52 
```
