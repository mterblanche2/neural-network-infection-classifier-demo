# Neural network: Infection classifier demo.
Demo of how to train and evaluate a simple deep neural network used to classify human infection data into clinically relevant categories.

## Background
I created this Jupyter Notebook as part of a teaching session showing medical staff how to train a basic neural network. The code is written in Python and I make extensive use of the Keras and Tensorflow libraries. The models are small and train easily, hence a GPU is not needed; I run the code on my MacBook.

While the train and test data resembles true values observed in patients, the data sets are entirely synthetic, i.e. were create using code rather than real patients.

## Requirements
- python 3.12

## Initial setup
- Set up an appropriate environment (e.g., .venv)
- Update pip: pip install --upgrade pip
- Install the requirements file: pip install -r requirements.txt

## Running the code
- This should be reasonably straightforward.
- In the CLI type "jupyter notebook" to open a browser. Click on "notebooks" then open "Deep Learning Teaching Demo.ipynb"
- The only thing to note is that details and build command for each model is in markup format and you need to change the cells' format to code while also changing the cells for the model you do not want to build to markup. This so they are not executed without you realising.
- Feel free to change the neurons and layers.

Feel free to use and modify, and do let me know if you spot any serious issues.

