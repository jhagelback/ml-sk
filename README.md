# Machine Learning examples
This is a collection of machine learning code examples in different languages and libraries.

## Jupyter kernels
For machine learning I prefer to code in Jupyter Notebooks (my current version is 5.7.4). There are additional Jupyter kernels available for other languages than Python, which is great if you wanna try out some stuff in Weka or R. I use the following kernels:
- IJava for Java (available [here](https://github.com/SpencerPark/IJava))
- IRKernel for R (available [here](https://irkernel.github.io/))

A list of Jupyter kernels can be found [here](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels). You can check which kernels you have installed with the command *jupyter kernelspec list*

## Folders

**Keras** contains examples using Keras (checked for version 2.2.5) with TensorFlow backend (checked for version 1.14.0).

**R** contains a few examples using R (checked for version 3.4.2). I am not a huge R fan, but have written some examples for students that prefer R.

**Weka** contains some examples using the Weka API (checked for version 3.8.3). The main benefit of Weka is its GUI, but learning how to use the API can sometimes be useful.

**sklearn** contains examples using Scikit-learn (checked for version 0.20.3). I usually prefer Scikit-learn for most tasks, along with Keras for deep learning.

**data** contains the datasets used in the code examples.

**images** contains example images used for the pre-trained image recognition example in the Keras folder.

## Run using Docker
1. Clone the GitHub repository to your computer
2. Run the Jupyter TensorFlow notebook and attach the notebooks directory by running the following command in a terminal:<br><code>docker run --rm -p 8888:8888 -v [put your local path here]/ml-sk:/home/jovyan jupyter/tensorflow-notebook</code>
3. Copy the URL you see in the terminal (<em>http://127.0.0.1:8888/?token=...</em>) and paste it in a web browser
4. Now you are ready to start hacking!

## Other Docker files
<b>R</b>:<br>
<code>docker run --rm -p 8888:8888 -v [put your local path here]/ml-sk:/home/jovyan jupyter/R-notebook</code><br>

Jupyter Notebook with R support.

<b>Java</b>:<br>
<code>docker run --rm -p 8888:8888 -v [put your local path here]/ml-sk:/home/jovyan jbindinga/java-notebook</code>

Jupyter Notebook with Java support.

<b>Jupyter extended</b>:<br>
<code>docker run --rm -p 8888:8888 -v [put your local path here]/ml-sk:/home/jovyan jhagelback/jupyter-ext</code>

Extended Jupyter Tensorflow notebook (available at <a href="https://hub.docker.com/r/jhagelback/jupyter-ext">Docker Hub</a>)  with some additional packages:
* XGBoost, OpenCV, mtcnn, gensim, pyod, import-ipynb

