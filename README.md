# Detection of Cardiac Arrhythmia

## What is the intent of this project?
Carrhythmia is a website which can help the doctors in determining whether the patient is suffering from Cardiac Arrhythmia or not. Detection of Cardiac Arrhythmia at an early stage can help avoid chronic heart diseases at a later stage. The system is developed using the Flask framework for deploying the ML model which, in this case, is the Xception model, a CNN architecture. Apart from detecting the disease, the system is also classifying it into the 14 classes of Arrhythmia with the help of Kernelized SVM. This system can prove to be super helpful to detect any cardiac irregularities at an early stage so as to avoid any major diseases or deaths in the future.

## How to setup the project on your local machine?
### Below are the steps to run the project on your local machine:

1. Create a virtual environment with the help of the following command  
``` python3 -m venv some-virtual-env-name ```
2. Enable the virtual environment by running the following command  
``` some-virtual-env-name\Scripts\activate ```
3. Clone the repository
4. Download all the required dependencies with the help of the below command  
``` pip install -r requirements.txt ```
5. Finally run the following command on your console and you're ready to go!  
``` python3 carrhythmia.py ```

## Website 
https://carrhythmia.herokuapp.com


## Paper Link
- [Application of Machine Learning in Cardiac Arrhythmia](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003133681-7/application-machine-learning-cardiac-arrhythmia-gresha-bhatia-shefali-athavale-yogita-bhatia-tanya-mohanani-akanksha-mittal?context=ubx&refId=d0e1a0d0-c3c3-45e5-a739-50b258375893)
- [Classification of Cardiac Arrhythmia using Kernelized SVM](https://ieeexplore.ieee.org/document/9143000)

