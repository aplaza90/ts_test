# Exercise Alvaro Plaza

## Content

The repository contains all the necessary code files to execute the pipeline,
a file with the code requirements, the train and predicted data, and a makefile to
set the environment up and run the pipeline.  

## Code Design

I have tried to design an open-to-extension code repository, making use of the
flexibility provided by object-oriented design.

Each forecasting model must conform to an abstract class (TimeSeriesModel) 
which will allow it to be used by the ModelSelector and to make predictions in
the pipeline without needing coupling.

Each phase of the pipeline makes use of functions and classes separated into
their own code file.

## Solution Design

The solution consists of a small pipeline that is executed in Python, consisting of:

- **Data reading and preprocessing:**
  - The data exploration phase was carried out with the functions of the module `data_exploration.py`. 
    It is worth noting that the exploration results indicated a low probability (p-value) of stationarity, as well as insignificant ACF and PACF.
  - Regarding preprocessing, it was not necessary to do anything more than adjustments for types and removal of nulls in data reading.

- **Model creation:**
  - For the solution, I chose two models representative of what can be a solution using
    classical methods to a problem with very few data: a SARIMA model and a handmade moving
    average model.
    
    I considered other models, but I prioritized making a flexible, comprehensible, and easily adaptable code.

- **Selection of the best model:**
  - This phase was carried out considering the minimum absolute error of the model with respect
    to the training data (the entire dataset).
    
    This choice is by no means rigorous, and again I wanted to prioritize the selection of a reasonable measure over speculations about the best metrics.

- **Forecasting data with the best model:**
  - The data were predicted according to the requirements.

## Testing

I used pytest and a dataset created ad hoc for designing unit tests that allowed 
me to work faster on code design, I have not included them as they are not part 
of the solution.


## Requirements

* Python 3.7
* venv (optional)
* make (optional)


## Local dev with venv (optional)

```sh
python3.7 -m venv .venv && source .venv/bin/activate

pip install requirements.txt
```


## Makefile

Contains a couple of command, useful to set up the environment
and execute the pipeline.


## Documentation

The code has been documented (when appropriated) using the Google style docstrings

## Conclusion
The code has certain design flaws and the solution is not rigorous at all. 
I have tried to adjust to the available time (a few hours).
Despite this, it is a model open to extension and the choice of forecasting
models is sufficiently conservative to allow incremental iterative development.



