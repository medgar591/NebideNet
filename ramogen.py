# RaMoGen
# Random Model Generator
# By Matt Edgar
# Verson 0.2

import numpy

# Generates a number of Simple Linear Regression models
# Attributes determines the number of coefficients of the models
# Models determines the number of models to generate
# fileName is where the models will be written - overwrites previous files
# gauss is a boolean to determine if Gaussian Normal Distribution (true) or Uniform (false) is used
def genSLR(attributes, models, fileName, gauss=True):
    text = [""]*models

    if gauss:
        for model in range(models):
            temp = numpy.random.normal(0,1,attributes)

            text[model] = ",".join(map(str,temp.tolist()))
            text[model] += "\n"
    else: # Note: numpy.random.uniform is exclusive of the highest value, which slightly shifts results
        for model in range(models):
            temp = numpy.random.uniform(-1,1,attributes)

            text[model] = ",".join(map(str,temp.tolist()))
            text[model] += "\n"
    
    result = open(fileName, "wt")
    result.writelines(text)
    result.close()

genSLR(4,3,"none")