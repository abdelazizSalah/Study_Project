# Task 3: Readme 
- For this task I needed to use these libraries:
  - argparse: used to parse the inputs from the user via command line. 
  - numpy: used to perform fast numerical computations on arrays and matricies. 
  - dataclasses: used to create simple classes structures and allow to overload built-in functions like __len__ to compute the length of the object from the class. 
  - enum: used to create enumeration for symbolic values specially used for determining static of dynamic fields. 
  - random: used for generating random values (used for creating a prototype before integrating Anna's work.)
  - typing: Used for type hints to make the code more easier to understand and more robus by specifying that a function accept a list of certain class

## How to install numpy
- the only external needed library here is numpy, and you can install it by running that command: 
  - > pip install -r requirments.txt

## How to run
- It would be better to run this task from the toolbox
- read the README.md file existing there first.
- To check how to use the toolbox use this command: 
  - > pyton task3.py -h
- Steps:
  - > cd Assignment2/Task3/
  - > python task3.py -s reverse_2 (no args then defualt max_len = 4 is used)
  - > python task3.py -s reverse_2 -m 6 (max_len = 6 is used)
- max_len here determine the maximum allowed length for a candidate keyword. 