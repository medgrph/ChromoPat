from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from Libs.features_extraction import HaralickFeatures, TAS, ZernikeMoments, extract, Preprocess, CenterMass, ChromatinFeatures
import pandas as pd

def featuresPreproc(preproc, data):
    
    if preproc=='minmax':
        scaler = preprocessing.MinMaxScaler()
    if preproc=='std':
        scaler = preprocessing.StandardScaler() 
    
    data = scaler.fit_transform(data)
    
    return data

def gridSearch(clf, parameters):

    cv = GridSearchCV(clf,parameters,cv=5)
    cv.fit(X_train,y_train)
    return cv

def dataCollection(img_dir, features, preproc_commands, normalize=False, enhance=False, nuclei_max_size=None):
    
    if preproc_commands is not None:
#         preprocess = Preprocess([dir1], max_size=nuclei_max_size)
        preprocess = Preprocess([img_dir], max_size=nuclei_max_size)

#     print('<========== Extraction of ' + str(features['object_1_class']) + ' features ==========>')
    data1 = extract(img_dir, features, preprocess, features['objects_1'], preproc_commands, enhance, normalize) 
#     print('<========== Extraction of ' + str(features['object_2_class']) + ' features ==========>')
#     data2 = extract(dir2, features, preprocess, features['objects_2'], 'object_2_class', preproc_commands, enhance, normalize)
    
    data = data1[:]
    df = pd.DataFrame(data=data)
    return df
#     df['class_codes'] =df['class_codes'].astype('category').cat.codes
#     df = shuffle(df, random_state=0)
#     df.reset_index(inplace=True, drop=True) 
#     X = df.drop(['class_codes'], axis=1)
#     y = df['class_codes']
    

# , X, y
    