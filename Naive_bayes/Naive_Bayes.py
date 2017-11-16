from clean_data import *

#Lazy learning
def predict_x(predictors, target, x):
    pro_vector = []
    Dataset = pd.concat([predictors, target], axis = 1)
    var, counts = np.unique(target, return_counts = True)
    fre = (counts.astype(float) + 1)/(len(target) + len(var))
    for labels, p in zip(var, fre):
        C_Dataset = Dataset[Dataset.iloc[:, -1] == labels]
        i = 0
        P_C = 1
        for feature in C_Dataset.columns:
            if i < len(x) - 1:
                sub_var, sub_counts = np.unique(C_Dataset[feature], return_counts = True)
                sub_fre = (sub_counts.astype(float)+1)/(len(C_Dataset[feature]) + len(sub_var))
                for k, v in zip(sub_var, sub_fre):
                    if x[i] == k:
                        P_C *= v
                i += 1
        h = p * P_C
        pro_vector.append(h)
    return var[np.argmax(pro_vector)]

def predict_test(testData):
    sum = 0
    len = 0
    for x in testData.T:
        if predict_x(predictors, target, testData.iloc[x, :]) == testData.iloc[x, 1]:
            sum += 1
        len += 1
    accurency = sum/len
    return accurency
        
        
            
             
                    
