import pandas as pd
import numpy as np
import scipy.stats.stats as st
import mlFeatures as ft
import csv

def computeFeatureVector(data):
    print "computeFeatureVector"
    stDeviation = np.std(np.array(data)) #Standard Deviation
    coVariant = st.variation(np.array(data)) # Coefficients of variation
    kurtosis = st.kurtosis(np.array(data)) #kurtosis
    features = ft.cFeatures(stDeviation, coVariant, kurtosis)
    return features

def createFeatureFromDataFile(file, classlabel):
    featuresData = []
    label = []
    df = pd.read_csv(file)  # df is DataFrame object
    x = df['X']
    y = df['Y']
    z = df['Z']
    i = 0;
    length = len(x)
    index = 0
    datasize = 400
    # print (df)
    while index < length:
        px = x[i:i+datasize]
        py = y[i:i+datasize]
        pz = z[i:i+datasize]
        i += datasize
        index += datasize
        featureX = computeFeatureVector(px)
        featureY = computeFeatureVector(py)
        featureZ = computeFeatureVector(pz)
        rowFeature = {'x' : featureX, 'y' : featureY, 'z' : featureZ}
        featuresData.append(rowFeature)
        label.append(classlabel)
        
    featureVector = {'features':featuresData, 'classlabel':label}
    return featureVector

def concateFeatureVector(featureVectorList):
    featureVectors = []
    classLabels = []
    for featureVector in featureVectorList:
        featureVectors = featureVectors + featureVector['features']
        classLabels = classLabels + featureVector['classlabel']
        
    return {'features':featureVectors, 'classlabel':classLabels}
        
def writeCSV(data):
    fieldNames = ['deviation_x', 'deviation_y','deviation_z','coefficient_x','coefficient_y','coefficient_z', 'kurtosis_x','kurtosis_y','kurtosis_z', 'classlabel'];
    with open('training_file.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter = ',', fieldnames=fieldNames)
        writer.writeheader()
        featureData = data['features']
        labelData = data['classlabel']
        dataLength = len(featureData)
        dataIndex = 0
        for feature in featureData:
            classLabel = labelData[dataIndex]
            featureX = feature['x']
            featureY = feature['y']
            featureZ = feature['z']
            row = {'deviation_x': featureX.deviation, 'deviation_y': featureY.deviation, 'deviation_z': featureZ.deviation,
                        'coefficient_x': featureX.coefficient, 'coefficient_y': featureY.coefficient, 'coefficient_z': featureZ.coefficient, 
                        'kurtosis_x': featureX.kurtosis, 'kurtosis_y': featureY.kurtosis, 'kurtosis_z': featureZ.kurtosis, 'classlabel': classLabel}
            writer.writerow(row)
            dataIndex +=1
            
    
if __name__ == "__main__":
    featureVectorList = []
    features1 = createFeatureFromDataFile('Hinton\Walking_1486450447311_10_Leg_nexus shakil_accelerometer.csv', 'walking')
    features2 = createFeatureFromDataFile('Hinton\Sitting_1486455964649_10_Leg_nexus nadim_accelerometer.csv', 'sitting')
    features3 = createFeatureFromDataFile('Hinton\Standing_1486448374458_10_Leg_nexus tahmid_accelerometer.csv', 'standing')
    featureVectorList.append(features1)
    featureVectorList.append(features2)
    featureVectorList.append(features3)
    concatedFeatures = concateFeatureVector(featureVectorList)
    writeCSV(concatedFeatures)
    print concatedFeatures