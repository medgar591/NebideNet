# RaMoGen
# Random Model Generator
# By Matt Edgar
# Verson 0.4

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
    else: # NOTE: numpy.random.uniform is exclusive of the highest value, which slightly shifts results
        for model in range(models):
            temp = numpy.random.uniform(-1,1,attributes)

            text[model] = ",".join(map(str,temp.tolist()))
            text[model] += "\n"
    
    result = open(fileName, "wt")
    result.writelines(text)
    result.close()


# Tests a file with models over a data set and appends the parity rating to each model
# modelFile is the file containing models as a list of vectors
# dataFile is the file containing the data to iterate over
# ignore is a list naming columns to ignore for various reasons, do not include the sensitive data column here, 0 indexed
# sensitive is the column containing the sensitive data, 0 indexed and assumed to be binary
def testSLR(modelFile, dataFile, ignore: list, sensitive):
    files = open(modelFile, "rt")
    models = [item.split(",") for item in files.read().splitlines()]
    files.close()

    models[:] = [ [float(num) for num in item] for item in models] #Turning strings into usable floats

    files = open(dataFile, "rt")
    data = [item.split(",") for item in files.read().splitlines()]
    files.close()

    data[:] = [ [float(num) for num in item] for item in data] # NOTE: This treats every data point as a float, even when some are binary

    # Delete ignored columns that are greater than the sensitive column
    ignore.sort(reverse=True) # When columns are removed from highest index to lowest, they don't mess with the next column deletion
    for column in [item > sensitive for item in ignore]: 
        for row in data:
            del row[column]
    
    # Pop sensitive column and convert to boolean
    sa = [bool(row.pop(sensitive)) for row in data]

    # Delete the rest of the columns
    for column in [item < sensitive for item in ignore]:
        for row in data:
            del row[column]
    

    #Scoring models
    lendata = len(data)
    for b in range(len(models)):
        score = [0.0]*lendata
        for x in range(lendata):
            score[x] = sum(
                [models[b][n+1] * data[x][n] for n in range(len(data[x]))],
                models[b][0])
            score[x] = max(0.0, min(score[x], 1.0)) # Effectively clamps results to a 0 to 1 scale
        
        #Calculating Statistical Parity
        st = sa.count(True) #Number of cases where Sensitive is True
        sf = len(sa) - st #Number of cases where Sensitive is False
        stt = 0.0 #Number of cases where Sensitive is True and score is True
        sft = 0.0 #Number of cases where Sensitive is False and score is True

        for n in range(len(sa)):
            if score[n] >= 0.5:
                if sa[n]:
                    stt += 1
                else:
                    sft += 1

        parity = (stt/st) - (sft/sf) #Parity = P(score+ | sensitive+) - P(score+ | sensitive-)
        models[b].append(parity)
    
    #Writing models + their parity scores back to the file
    for item in range(len(models)):
        models[item] = ",".join(map(str, models[item]))
        models[item] += "\n"
    files = open(modelFile, "wt")
    files.writelines(models)
    files.close()

# genSLR(100,1000,"slrmodels.csv") #Generates 1,000 sample models to test on the communitycrime dataset
testSLR("slrmodels.csv", "communitycrime/crimecommunity.csv", [100, 101], 0)