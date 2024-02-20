# AMR Problem Set

## Introduction

Data for this challenge is available in `data/`. You will find two `.csv` files:

* `admissions.csv`
* `patients.csv`

`admissions.csv` has records of patients' admissions to hospital, specifically their AMR results. `patients.csv` has information on the patients. 

## General instructions

Please do not fork this repository or interact with it directly. 

## Data dictionary

### Admissions

* `patient_id`: unique identifier for patients
* `date_admission`: the date of admission. This has already been standardised for you so that `1` is the first admission for the patient. The exact date is not provided as it is not required for this problem.
* `ciprofloxacin`: result for ciprofloxacin antibiotic. `S` is sensitive, which is good as that means doctors can use this antibiotic to treat the patient. `R` is resistant, signifying AMR. Doctors should not use drugs that are `R` to treat patients. 
* `gentamicin`: result for gentamicin antibiotic
* `amoxicillin_clavulanic_acid`:  result for amoxicillin-clavulanic acid

### Patients

* `patient_nos`: unique patient identifiers (links to `patient_id`)
* `sex`: patient sex
* `age`: patient age

## Tasks

These files contain AMR data that can provide insight for doctors treating patients. Please perform the following tasks:

1) Descriptive data analysis of patients and antibiotic resistance.
1) Create a model of your choice to predict AMR (i.e., antibiotics which are `R`). The specifics of your model are up to you, it could be a frequentist approach, bayesian approach, or machine learning approach. 'Black box' approaches are also fine. 
1) Develop an algorithm that determines whether each patient has had AMR to a particular antibiotic at any point in the past. For example:

        ```
        ##    patient_id date  ciprofloxacin ciprofloxacin_previous_R
        ##  1 10038332   1          S                FALSE               
        ##  2 10038332   2          S                FALSE               
        ##  3 10038332   3          S                FALSE               
        ##  4 10038332   4          R                FALSE               
        ##  5 10038332   5          R                TRUE                
        ##  6 10038332   6          S                TRUE                
        ##  7 10016742   1          S                FALSE               
        ##  8 10016742   2          R                FALSE               
        ##  9 10016742   3          S                TRUE               
        ## 10 10016742   4          R                TRUE               
        ## 11 10016742   5          R                TRUE
        ```

1) Create a function, called `previous_resistance` that accepts hospital number and antibiotic name as an input, and returns `TRUE` if patient has had antibiotic resistance to a specific antibiotic (provided by antibiotic name argument) at any time point within the dataset. Function should return `FALSE` if the patient does not have resistance to the antibiotic, or if the patient is not in the dataset (we assume they do not have resistance in this case).
1) Use your model and `previous_resistance` function and create a dashboard/web frontend for a doctor to use. The doctor should input `patient_id` and an antibiotic name (ciprofloxacin, gentamicin or amoxicillin-clavulanic acid), and the algorithm should do the following:
    * if the patient has resistance within the dataset to the antibiotic, then it should alert the doctor of this
    * otherwise, if resistance to the agent is not present in the dataset, or the patient is not in the dataset, the algorithm should use the prediction model from above to predict the risk of `R` for the antibiotic, and return this prediction to the doctor. 

## Hints

Please try to keep functions/classes modular, in reality datasets would have many more antibiotics, so we need to make sure code is generalisable. 

You can use any frameworks, modules or programming languages of your choice. We would like to see the code. 

Consider unit testing of the `previous_resistance` function, and any other functions/classes as appropriate.
