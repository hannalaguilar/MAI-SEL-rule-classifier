# CN2 - MAI - SEL
The aim of this project is to implement and validate 
the CN2 algorithm on three different datasets using Python.
The CN2 algorithm is a rule-based classifier that induce 
rules from the training data and display the 
rules in an interpretable way.

**The final report is in the docs folder.**

## How tu run the code

### Create an environment and activate it:
 ```
conda create --name cn2-py310 python=3.10
conda activate cn2-py310
pip install -r requirements.txt
 ```

### Run main.py:
 ```
python src/main.py
 ```

It will showcase the rule list, training accuracy, testing accuracy, 
and training time for the Iris and Titanic datasets. 
However, it should be noted that the Obesity dataset has been excluded
from this list due to its lengthy runtime.