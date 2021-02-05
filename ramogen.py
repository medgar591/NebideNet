# RaMoGen
# Random Model Generator
# By Matt Edgar
# Verson 0.1

import random

# Generates a number of Simple-Linear-Regression models based on inputs values
# Input values should be tuned to closely capture potential data ranges, as the slope will not be higher/lower than their ranges
# ihigh and ilow are used to generate a range of independent variable values
# dhigh and dlow are used similarly for the dependent variable
# models stands for how many models should be generated
# fileName points toward an existing file to be overwritten or nothing if a new file is to be made
def genSLR(ilow, ihigh, dlow, dhigh, models, fileName):

    text = [""]*models

    drange = dhigh-dlow
    irange = ihigh-ilow

    for model in range(len(text)):
        slope = drange/irange
        b1 = random.uniform(-slope, slope)
        
        # Given the slope b1, shift the line vertically so that the intercept is between either the low corner or the high corner
        if b1 >= 0:
            b0 = random.uniform(dlow-b1*ilow, dhigh-b1*ihigh)
        else:
            b0 = random.uniform(dhigh-b1*ilow, dlow-b1*ihigh)

        text[model] = str(b0) + "," + str(b1) + "\n"

    result = open(fileName, "wt")
    
    result.writelines(text)

    result.close()

genSLR(2,5,0,1,1000,"slrmodels.csv")