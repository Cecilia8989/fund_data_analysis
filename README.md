<h1 align="center">Fundamental for Data Analysis  - Tasks and Project</h1>
<h1 align="center">Higher Diploma in Computer Science with Data Analytics</h1>
   <p align="center">
   Cecilia Pastore 

## Description

This repository contains the work required for the subject "Fundamentals for Data Analysis," part of the Higher Diploma in Computer Science with Data Analytics at Atlantic Technological University.

## Repository Structure

It consists of two main files:

1. **tasks.ipynb**: A Jupyter notebook containing solutions to five tasks provided during lectures. These tasks cover various topics of interest discussed in the class.

<details>
    <summary> Tasks Assignement </summary>
           <p>

1. The Collatz conjecture1 is a famous unsolved problem in mathematics. The problem is to prove that if you start with any positive integer x and repeatedly apply the function f(x) below, you always get stuck in the repeating sequence 1, 4, 2, 1, 4, 2, . . .
$$
 f(n) =
  \begin{cases}
    n/2       & \quad \text{if } n \text{ is even}\\
    -(n+1)/2  & \quad \text{if } n \text{ is odd}
  \end{cases}
$$
For example, starting with the value 10, which is an even number, we divide it by 2 to get 5. Then 5 is an odd number so, we multiply by 3 and add 1 to get 16. Then we repeatedly divide by 2 to get 8, 4, 2, 1. Once we are at 1, we go back to 4 and get stuck in the repeating sequence 4, 2, 1 as we suspected. Your task is to verify, using Python, that the conjecture is true for the first 10,000 positive integers.
2. Give an overview of the famous penguins data set, explaining the types of variables it contains. Suggest the types of variables that should be used to model them in Python, explaining your rationale.
3. For each of the variables in the penguins data set,3 suggest what 3 mwaskom/seaborn-data: Data repository for seaborn examples. probability distribution from the numpy random distributions list is the most appropriate to model the variable.
4. Suppose you are flipping two coins, each with a probability p of giving heads. Plot the entropy of the total number of heads versus p.
5. Create an appropriate individual plot for each of the variables in the penguin data set.

</p>
</details>

2. **project.ipnyb**: Another Jupyter notebook that includes the final project for the subject. The project involves an analysis of the notorious Iris dataset.

</p>
</details>

<details>
    <summary> Project Assignement: </summary>
           <p>

The project is to create a notebook investigating the variables and data points within the well-known iris flower data set associated with Ronald A Fisher.
• In the notebook, you should discuss the classification of each variable within the data set according to common variable types and scales of measurement in mathematics, statistics, and Python.
• Select, demonstrate, and explain the most appropriate summary statistics to describe each variable.
• Select, demonstrate, and explain the most appropriate plot(s) for each variable.
• The notebook should follow a cohesive narrative about the data set

</p>
</details>

## Technology used 

This project have been writen using python 3.11.4 and structurate in a Jupyter notebook using Visual code, version 1.85.1, as an interpreter. Consequently, both the file **tasks.ipynb and project.ipynb** can be run with  jupyter nootebook or can be open with VS Code. 

## Running the script.

1. Make sure python is installed on your machine. If not [Anaconda](https://www.anaconda.com/) can be installed.

2. Install Jupyter Notebook. Jupyter notebook [can be installed in a terminal](https://jupyter.org/install) tapying on the terminal:

```python
pip install notebook
```
3. I advice also to install a specific extension of jupyter notebook: [jupyter-contrib-nbextensions ](https://pypi.org/project/jupyter-contrib-nbextensions/). This can be done typing on the terminal:

```python
pip install jupyter_contrib_nbextensions
```

3. Install the  needed libraries.

The script use several python package and library and those need to be installed on the system.

The library needed are:
- pandas
- numpy
- seaborn 
- matplotlib
- plotly
- scikit-learn
- scipy
- io 

To install them you can use pip install in a terminal, or command prompt, followed by the librarys:

```python
pip install pandas 
pip install matplotlib 
pip install seaborn
pip install plotly
pip install scikit-learn
pip install scipy
pip install io
```
4. Clone the repository in a local machine using [git clone](https://robots.net/how-to-guide/how-to-download-a-github-repository/).

5. Open the files **tasks.ipynb**/**project.ipynb** in VS Code or, in a terminal, navigate to the directory where the file is saved and run jupyter notebook with the following code:

```python
jupyter notebook
```
---