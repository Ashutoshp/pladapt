import sys
#from collections import defaultdict
#from ProblemDatabase import ProblemFeatures
#from ProblemDatabase import ProblemDatabase
#from ProblemDatabase import ProblemInstance
from sklearn.preprocessing import StandardScaler
from DartDBWrapper import DBWrapper
#from ProblemDatabase import *
#from ProblemDB import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
import random
from collections import Counter
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
#from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import PCA
from statistics import *
import matplotlib.pyplot as plt
import pandas as pd

#import numpy

def scatterplot(x_data, y_data, x_label="data", y_label="label", title="", color = "r", yscale_log=False):

    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def get_random_numbers(minimum, maximum, count):
    random_numbers = set()
        
    for x in range(count):
        random_numbers.add(random.randint(minimum, maximum))

    return random_numbers

def get_n_random_numbers(minimum, maximum, count):
    random_numbers = set()
        
    while (len(random_numbers) < count):
        random_numbers.add(random.randint(minimum, maximum))

    return random_numbers

class CrossValidate:
    #__slots__ = ['__features_list', '__labels_list', '__problems']
    
    def __init__(self, data_file, fold, ignored_seeds = "", debug_file = "", cv_file = ""):
        self.__data_file = data_file
        self.__fold = int(fold)
        self.__db = DBWrapper(data_file, ignored_seeds)
        self.__ignored_seeds = ignored_seeds
        #self.__debug_file_name = str(fold) + "_ExtraTreesClassifier_log.csv"
        self.__file = debug_file
        #self.__cv_file_name = str(fold) + "_cv_ExtraTreesClassifier_log.csv"
        self.__cv_file = cv_file
        self.__features_list = ""
        self.__labels_list = ""
        #open("cv_log", "w")
 
    def get_all_traces(self):
        return self.__db.get_all_traces()

    def cv_per_trace(self):
        print("############ ERROR: cv_per_trace NOT IMPLEMENTED ##############")
        assert False
        return 0, 0

    def test_pca(self, features, labels):
        features_list = StandardScaler().fit_transform(features)
       
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(features_list)
        #print(principalComponents)
        #principalDf = pd.DataFrame(data = principalComponents, \
        #        columns = ['principal component 1', 'principal component 2'])
        #df = pd.Series(labels)
        #finalDf = pd.concat([principalDf, df], axis = 1)

        #print(finalDf)
        
        #print(finalDf[2])
        #print(finalDf[0])

        #fig = plt.figure(figsize = (8,8))
        #ax = fig.add_subplot(1,1,1) 
        #ax.set_xlabel('Principal Component 1', fontsize = 15)
        #ax.set_ylabel('Principal Component 2', fontsize = 15)
        #ax.set_title('2 component PCA', fontsize = 20)
        #targets = [0, 2, 1]
        #colors = ['r', 'g', 'b']
        #for target, color in zip(targets,colors):
        #    indicesToKeep = finalDf[0] == target
        #    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'] \
        #       , finalDf.loc[indicesToKeep, 'principal component 2'] \
        #       , c = color \
        #       , s = 50)
        #ax.legend(targets)
        #ax.grid()
        #plt.show()

        return principalComponents, labels

    def cv_per_problem(self):
        all_training_problems = self.__db.get_all_training_problems()
        scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', \
                    'precision_macro', 'recall_macro', 'f1_macro', \
                    'precision_weighted', 'recall_weighted', 'f1_weighted']
        #X_resampled, y_resampled = SMOTE().fit_resample(all_training_problems.get_features_list(), \
        #        all_training_problems.get_label_list())
        #self.test_pca(all_training_problems.get_features_list(), all_training_problems.get_label_list())
        features_list = StandardScaler().fit_transform(all_training_problems.get_features_list())
        X_resampled, y_resampled = ADASYN().fit_resample(features_list, \
                all_training_problems.get_label_list())
        #X_resampled, y_resampled = self.test_pca(X_resampled, y_resampled)

        #X_resampled, y_resampled = BorderlineSMOTE().fit_resample(all_training_problems.get_features_list(), \
        #            all_training_problems.get_label_list())

        #plt.plot(y_resampled)
        #plt.show()
        #plt.plot(all_training_problems.get_label_list())
        #plt.show()
        #scatterplot(all_training_problems.get_features_list(), all_training_problems.get_label_list())
        #scatterplot(X_resampled, y_resampled)
        #smote_file = open("data.csv", "w")
        #index = 0
        #features_list = all_training_problems.get_features_list()
        #labels_list = all_training_problems.get_label_list()
        #while (index < len(features_list)):
        #    featrs = features_list[index]
        #    for f in featrs:
        #        smote_file.write(str(f))
        #        smote_file.write(', ')
        #    smote_file.write(str(labels_list[index]))
        #    smote_file.write('\n')
        #    index = index + 1

        #smote_file.close()

        #smote_file = open("smote.csv", "w")
        #index = 0
        #while (index < len(y_resampled)):
        #    featrs = X_resampled[index]
        #    for f in featrs:
        #        smote_file.write(str(f))
        #        smote_file.write(', ')
        #    smote_file.write(str(y_resampled[index]))
        #    smote_file.write('\n')
        #    index = index + 1

        #smote_file.close()
        #print("len(X_resampled) = ", len(X_resampled))
        print("len(y_resampled) = ", len(y_resampled))
        print(sorted(Counter(y_resampled).items()))
        #clf = svm.SVC(kernel='linear', C=0.5)
        #clf = LinearSVC(random_state=0, C=0.000001) 
        #multi_class='crammer_singer')
        #clf = svm.SVC(C=500, decision_function_shape='ovr')
        #clf = KNeighborsClassifier(n_neighbors = 4, algorithm = 'ball_tree')
        #clf = RadiusNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
        #clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')
        #clf = LogisticRegression(C=1.0, solver='sag', multi_class='multinomial')
        #clf = LogisticRegression(solver='sag')
        #clf = LogisticRegression(solver='saga')
        #clf = LogisticRegression(solver='lbfgs')
        #clf = LogisticRegressionCV(solver='newton-cg')
        #clf = MLPClassifier(solver='sgd') # precision/recall for label 0 is always 0
        #clf = DecisionTreeClassifier(random_state=0, max_depth=100, max_features=None, class_weight='balanced')
        #clf = DecisionTreeClassifier(random_state=0, max_depth=100,max_features=None, class_weight= {0:0.2, 1:.5, 2:0.3})
        #clf = RandomForestClassifier(max_depth=50, random_state=0)
        #clf = ExtraTreesClassifier(n_estimators=90)
        #clf = AdaBoostClassifier()
        #clf = GradientBoostingClassifier(max_depth=3) # Reasonbale precision for 0, recall/precision for 2
        clf = BaggingClassifier(n_estimators=20)

        #n_estimators = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        #criterion = ['gini', 'entropy']
        #max_depth = [None, 20, 30, 50]
        #min_samples_split = [2, 12, 22]
        #max_features = [None, 'log2', 'sqrt', 'auto']

        #n_estimators = [80, 83, 85, 87, 90, 91, 93, 95, 97, 100, 102, 105, 107, 110]
        #criterion = ['gini']
        #max_depth = [None]
        #min_samples_split = [2]
        #max_features = ['auto']
        #loss = ['deviance']

        # ExtraTree
        #self.__file.write("n_estimators, criterion, max_depth, min_samples_split, max_features, \
        #                recall_score_0, recall_score_1, recall_score_2, precision_score_0, precision_score_1, precision_score_2, \
        #                f1_score_0, f1_score_1, f1_score_2, recall_score_macro, recall_score_micro, precision_score_macro, \
        #                precision_score_micro, f1_score_macro, f1_score_micro\n")
        #self.__file.write("n_estimators, loss, max_depth, min_samples_split, max_features, \
        #                recall_score_0, recall_score_1, recall_score_2, precision_score_0, precision_score_1, precision_score_2, \
        #                f1_score_0, f1_score_1, f1_score_2, recall_score_macro, recall_score_micro, precision_score_macro, \
        #                precision_score_micro, f1_score_macro, f1_score_micro\n")
        #self.__cv_file.write("n_estimators, criterion, max_depth, min_samples_split, max_features, \
        #                recall_0_avg, recall_0_dev, recall_1_avg, recall_1_dev, recall_2_avg, recall_2_dev,\
        #                recall_macro, recall_micro, recall_weighted, precision_0_avg, precision_0_dev, \
        #                precision_1_avg, precision_1_dev, precision_2_avg, precision_2_dev, \
        #                precision_macro, precision_micro, precision_weighted, f1_0_avg, f1_0_dev, \
        #                f1_1_avg, f1_1_dev, f1_2_avg, f1_2_dev, f1_macro, f1_micro, f1_weighted\n")

        #GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, \
                #subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, \
                #min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, \
                #min_impurity_split=None, init=None, random_state=None, max_features=None, \
                #verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’)
        #max_recall_0_mean = 0
        #best_estimator_count = 10 # Start with the default value

       #for estimator in n_estimators:
       #     for c in criterion:
       #         for depth in max_depth:
       #             for sample_split in min_samples_split:
       #                 for features in max_features:
       #                     recall_0 = []
       #                     recall_1 = []
       #                     recall_2 = []

       #                     precision_0 = []
       #                     precision_1 = []
       #                     precision_2 = []

       #                     f1_0 = []
       #                     f1_1 = []
       #                     f1_2 = []

       #                     clf = ExtraTreesClassifier(n_estimators=estimator, criterion=c, \
       #                             max_depth=depth, min_samples_split=sample_split, \
       #                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
       #                             max_features=features, max_leaf_nodes=None, \
       #                             min_impurity_decrease=0.0, min_impurity_split=None, \
       #                             bootstrap=False, oob_score=False, \
       #                             n_jobs=1, random_state=None, verbose=0, \
       #                             warm_start=False, class_weight=None)
        
                            #clf = GradientBoostingClassifier(loss=l, learning_rate=0.1, n_estimators=estimator, \
                            #        subsample=1.0, criterion='friedman_mse', min_samples_split=sample_split, min_samples_leaf=1, \
                            #        min_weight_fraction_leaf=0.0, max_depth=depth, min_impurity_decrease=0.0, \
                            #        min_impurity_split=None, init=None, random_state=None, max_features=features, \
                            #        verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        skf = StratifiedKFold(n_splits=self.__fold)
        features_list = X_resampled
        labels_list = y_resampled
        skf.get_n_splits(features_list, y_resampled)
        
        #skf.get_n_splits(all_training_problems.get_features_list(), \
        #        all_training_problems.get_label_list())
        #features_list = all_training_problems.get_features_list()
        #labels_list = all_training_problems.get_label_list()
        
        fold = 0
        recall_0 = []
        recall_1 = []
        recall_2 = []

        precision_0 = []
        precision_1 = []
        precision_2 = []

        f1_0 = []
        f1_1 = []
        f1_2 = []

        for train_index, test_index in skf.split(features_list, labels_list):
            #print("TRAIN:", train_index, "TEST:", test_index)

            tr_features = list()
            tr_labels = list()
            tt_features = list()
            tt_labels = list()

            for i in train_index:
                tr_features.append(features_list[i])
                tr_labels.append(labels_list[i])

            for i in test_index:
                tt_features.append(features_list[i])
                tt_labels.append(labels_list[i])

            #print(tr_features)
            clf.fit(tr_features, tr_labels)
            predicted_labels = list()
            train_predicted_labels = list()

                                #for problem in tr_features:
                                #    predicted_label = clf.predict([problem])
                                #    train_predicted_labels.append(predicted_label)

                                #i = 0
                                #while i < len(train_predicted_labels):
                                    #print("Predicted,  Actual ", train_predicted_labels[i], tr_labels[i])
                                #    i = i + 1

                                #print("Confusion Matrix for train data = ", confusion_matrix(tr_labels, train_predicted_labels))

            for problem in tt_features:
                #print(problem)
                predicted_label = clf.predict([problem])
                predicted_labels.append(predicted_label)

            i = 0

            #while i < len(predicted_labels):
            #    print("Predicted,  Actual ", predicted_labels[i], tt_labels[i])
            #    i = i + 1

            print("&&&&&&&&&& Fold = ", fold, "&&&&&&&&&&")
            fold = fold + 1
            print("Confusion Matrix = ", confusion_matrix(tt_labels, predicted_labels))
            
            recall = recall_score(tt_labels, predicted_labels, average=None)
            print("recall_score = ", recall)

            prec_score = precision_score(tt_labels, predicted_labels, average=None)
            print("precision_score = ", prec_score)
                               
            f1 = f1_score(tt_labels, predicted_labels, average=None)
            print("f1_score = ", f1)
                                   
            recall_macro = recall_score(tt_labels, predicted_labels, average='macro')
            print("recall_score_macro = ", recall_macro)
                                
            recall_micro = recall_score(tt_labels, predicted_labels, average='micro')
            print("recall_score_micro = ", recall_micro)
                                
            precision_macro = precision_score(tt_labels, predicted_labels, average='macro')    
            print("precision_score_macro = ", precision_macro)
                                    
            precision_micro = precision_score(tt_labels, predicted_labels, average='micro')    
            print("precision_score_micro = ", precision_micro)
                                   
            f1_macro = f1_score(tt_labels, predicted_labels, average='macro') 
            print("f1_score_macro = ", f1_macro)
                                    
            f1_micro = f1_score(tt_labels, predicted_labels, average='micro')
            print("f1_score_micro = ", f1_micro)
            
            print("&&&&&&&&&&&&&&&&&&&&&&&&&")

            recall_0.append(recall[0])
            recall_1.append(recall[1])
            recall_2.append(recall[2])

            precision_0.append(prec_score[0])
            precision_1.append(prec_score[1])
            precision_2.append(prec_score[2])

            f1_0.append(f1[0])
            f1_1.append(f1[1])
            f1_2.append(f1[2])
                                #row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                #    + str(sample_split) + ", " + str(features) + ", " \
                                #    + str(recall[0]) + ", " + str(recall[1])  + ", " + str(recall[2]) + ", " \
                                #    + str(prec_score[0]) + ", " + str(prec_score[1])  + ", " + str(prec_score[2]) + ", " \
                                #    + str(f1[0]) + ", " + str(f1[1])  + ", " + str(f1[2]) + ", " \
                                #    + str(recall_macro) + ", " + str(recall_micro) + ", " + str(precision_macro) + ", " \
                                #    + str(precision_micro) + ", " + str(f1_macro) + ", " + str(f1_micro) +  '\n'

                                #row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                #    + str(sample_split) + ", " + str(features) + ", " \
                                #    + str(recall[0]) + ", " + str(recall[1])  + ", " + str(recall[2]) + ", " \
                                #    + str(prec_score[0]) + ", " + str(prec_score[1])  + ", " + str(prec_score[2]) + ", " \
                                #    + str(f1[0]) + ", " + str(f1[1])  + ", " + str(f1[2]) + ", " \
                                #    + str(recall_macro) + ", " + str(recall_micro) + ", " + str(precision_macro) + ", " \
                                #    + str(precision_micro) + ", " + str(f1_macro) + ", " + str(f1_micro) +  '\n'
                                #self.__file.write(row)
                                #self.__cv_file.write(row)

        print("calling cross_validate")
        scores = cross_validate(clf, features_list, labels_list, scoring=scoring, cv=skf, \
                return_train_score=True)
                            #print("score keys = ", sorted(scores.keys()))
                            #print("f1_score_micro = ", scores['test_f1_micro'])
                            #print(str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                            #        + str(sample_split) + ", " + str(features))
        print("*******************************************************")
        recall_0_mean = mean(recall_0)
        print("recall_0_mean = ", recall_0_mean)
            #recall_0_dev = stdev(recall_0)
            #print("recall_0_dev = ", recall_0_dev)
                            
        recall_1_mean = mean(recall_1)
        print("recall_1_mean = ", recall_1_mean)
            #recall_1_dev = stdev(recall_1)
            #print("recall_1_dev = ", recall_1_dev)
                            
        recall_2_mean = mean(recall_2)
        print("recall_2_mean = ", recall_2_mean)
            #recall_2_dev = stdev(recall_2)
            #print("recall_2_dev = ", recall_2_dev)
                            
            #print("recall_macro = ", scores['test_recall_macro'])
        recall_macro_mean = scores['test_recall_macro'].mean()
        print("recall_macro_mean = ", recall_macro_mean)
            #print("recall_macro_std = ", scores['test_recall_macro'].std())

            #print("recall_micro = ", scores['test_recall_micro'])
        recall_micro_mean = scores['test_recall_micro'].mean()
        print("recall_micro_mean = ", recall_micro_mean)
            #print("recall_micro_std = ", scores['test_recall_micro'].std())
        
            #print("recall_weighted = ", scores['test_recall_weighted'])
        recall_weighted_mean = scores['test_recall_weighted'].mean()
        print("recall_weighted_mean = ", recall_weighted_mean)
            #print("recall_weighted_std = ", scores['test_recall_weighted'].std())
                            
        print("#########################")
        precision_0_mean = mean(precision_0)
        print("precision_0_mean = ", precision_0_mean)
            #precision_0_dev = stdev(precision_0)
            #print("precision_0_dev = ", precision_0_dev)
                            
        precision_1_mean = mean(precision_1)
        print("precision_1_mean = ", precision_1_mean)
            #precision_1_dev = stdev(precision_1)
            #print("precision_1_dev = ", precision_1_dev)
                            
        precision_2_mean = mean(precision_2)
        print("precision_2_mean = ", precision_2_mean)
            #precision_2_dev = stdev(precision_2)
            #print("precision_2_dev = ", precision_2_dev)
                            
            #print("precision_macro = ", scores['test_precision_macro'].mean())
        precision_macro_mean = scores['test_precision_macro'].mean()
        print("precision_macro_mean = ", precision_macro_mean)
            #print("precision_macro_std = ", scores['test_precision_macro'].std())
        
            #print("precision_micro = ", scores['test_precision_micro'].mean())
        precision_micro_mean = scores['test_precision_micro'].mean()
        print("precision_micro_mean = ", precision_micro_mean)
            #print("precision_micro_std = ", scores['test_precision_micro'].std())
        
            #print("precision_weighted = ", scores['test_precision_weighted'])
        precision_weighted_mean = scores['test_precision_weighted'].mean()
        print("precision_weighted_mean = ", precision_weighted_mean)
            #print("precision_weighted_std = ", scores['test_precision_weighted'].std())
        print("#########################")
        f1_0_mean = mean(f1_0)
        print("f1_0_mean = ", f1_0_mean)
            #f1_0_dev = stdev(f1_0)
            #print("f1_0_dev = ", f1_0_dev)
            
        f1_1_mean = mean(f1_1)
        print("f1_1_mean = ", f1_1_mean)
            #f1_1_dev = stdev(f1_1)
            #print("f1_1_dev = ", f1_1_dev)
            
        f1_2_mean = mean(f1_2)
        print("f1_2_mean = ", f1_2_mean)
            #f1_2_dev = stdev(f1_2)
            #print("f1_2_dev = ", f1_2_dev)
            
        f1_score_macro_mean = (scores['test_f1_macro']).mean()
        print("f1_score_macro mean = ", f1_score_macro_mean)
            #print("f1_score_macro std = ", (scores['test_f1_macro']).std())
        f1_score_micro_mean = (scores['test_f1_micro']).mean()
        print("f1_score_micro mean = ", f1_score_micro_mean)
            #print("f1_score_micro std = ", (scores['test_f1_micro']).std())
        
        f1_score_weighted_mean = scores['test_f1_weighted'].mean()
        print("f1_score_weighted_mean = ", f1_score_weighted_mean)
            #print("f1_score_weighted_std = ", scores['test_f1_weighted'].std())
        print("*******************************************************")

            #if (max_recall_0_mean < recall_0_mean):
            #    max_recall_0_mean = recall_0_mean
            #    best_estimator_count = estimator

                            #cv_row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                            #        + str(sample_split) + ", " + str(features) + ", " \
                            #        + str(recall_0_mean) + ", " + str(recall_0_dev) + ", " \
                            #        + str(recall_1_mean) + ", " + str(recall_1_dev) + ", " \
                            #        + str(recall_2_mean) + ", " + str(recall_2_dev) + ", " \
                            #        + str(recall_macro_mean) + ", " + str(recall_micro_mean) + ", " \
                            #        + str(recall_weighted_mean) + ", " \
                            #        + str(precision_0_mean) + ", " + str(precision_0_dev) + ", " \
                            #        + str(precision_1_mean) + ", " + str(precision_1_dev) + ", " \
                            #        + str(precision_2_mean) + ", " + str(precision_2_dev) + ", " \
                            #        + str(precision_macro_mean) + ", " \
                            #        + str(precision_micro_mean) + ", " + str(precision_weighted_mean) + ", " \
                            #        + str(f1_0_mean) + ", " + str(f1_0_dev) + ", " \
                            #        + str(f1_1_mean) + ", " + str(f1_1_dev) + ", " \
                            #        + str(f1_2_mean) + ", " + str(f1_2_dev) + ", " \
                            #        + str(f1_score_macro_mean) + ", " + str(f1_score_micro_mean) + ", " \
                            #        + str(f1_score_weighted_mean) + '\n'
                            
                            #self.__cv_file.write(cv_row)
        #print("Score", scores)

        #print("f1_score_samples = ", scores['test_f1_samples'])
        #print("precision_samples = ", scores['test_precision_samples'])
        #print("recall_samples = ", scores['test_recall_samples'])
        #print("accuracy = ", scores['test_accuracy'])

        print("precision_0_mean = ", precision_0_mean)
        print("precision_1_mean = ", precision_1_mean)
        print("precision_2_mean = ", precision_2_mean)
        print("recall_0_mean = ", recall_0_mean)
        print("recall_1_mean = ", recall_1_mean)
        print("recall_2_mean = ", recall_2_mean)
        return recall_0_mean
        
    def read_file(self):
        self.__db.read_db()

    def test_ignored_seeds(self, merge_labels, estimators):
        #ExtraTreesClassifier(n_estimators=10, criterion=’gini’, max_depth=None, \
        #min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
        #max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, \
        #min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, \
        #random_state=None, verbose=0, warm_start=False, class_weight=None)
        clf = ExtraTreesClassifier(n_estimators = estimators)
        clf.fit(self.__features_list, self.__labels_list)

        all_ignored_problems = self.__db.get_ignored_seed_problems()
        problem_features = all_ignored_problems.get_features_list()
        problem_labels = all_ignored_problems.get_label_list()
        predicted_labels = list()

        for problem in problem_features:
            #print("problem = ", problem)
            predicted_label = clf.predict([problem])
            predicted_labels.append(predicted_label[0])

        #print("predicted_labels = ", predicted_labels)
        #print("problem_labels = ", problem_labels) 

        if (merge_labels):
            print("%%%%%%%%%%%% Merging Labels %%%%%%%%%%")
            i = 0
            while i < len(predicted_labels):
                if (predicted_labels[i] == 2):
                    #print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                    predicted_labels[i] = 0
                    #print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])

                if (problem_labels[i] == 2):
                    #print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                    problem_labels[i] = 0
                    #print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])
            
                #if (problem_labels[i] == 0 and predicted_labels[i] == 2):
                #    print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                #    predicted_labels[i] =[0]
                #    print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])

                #if (problem_labels[i] == 1 and predicted_labels[i] == 2):
                #    print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                #    predicted_labels[i] = [0]
                #    print("### Actual, Predicted ", problem_labels[i], predicted_labels[i])
                i = i + 1

        #print("predicted_labels = ", predicted_labels)
        #print("problem_labels = ", problem_labels) 
        
        print("Confusion Matrix = ", confusion_matrix(problem_labels, predicted_labels))
            
        recall = recall_score(problem_labels, predicted_labels, average=None)
        print("recall_score = ", recall)

        prec_score = precision_score(problem_labels, predicted_labels, average=None)
        print("precision_score = ", prec_score)
       
        f1 = f1_score(problem_labels, predicted_labels, average=None)
        print("f1_score = ", f1)
        
        recall_macro = recall_score(problem_labels, predicted_labels, average='macro')
        print("recall_score_macro = ", recall_macro)
                                
        recall_micro = recall_score(problem_labels, predicted_labels, average='micro')
        print("recall_score_micro = ", recall_micro)
                                
        precision_macro = precision_score(problem_labels, predicted_labels, average='macro')    
        print("precision_score_macro = ", precision_macro)
                                    
        precision_micro = precision_score(problem_labels, predicted_labels, average='micro')    
        print("precision_score_micro = ", precision_micro)
                                   
        f1_macro = f1_score(problem_labels, predicted_labels, average='macro') 
        print("f1_score_macro = ", f1_macro)
                                    
        f1_micro = f1_score(problem_labels, predicted_labels, average='micro')
        print("f1_score_micro = ", f1_micro)
        print("&&&&&&&&&&&&&&&&&&&&&&&&&")
        
        label_0_present = 0 in problem_labels
        label_1_present = 1 in problem_labels
        label_0_predicted = 0 in predicted_labels
        label_1_predicted = 1 in predicted_labels

        metrics = ['inf', 'inf', 'inf', 'inf']
        
        if (label_0_present):
            metrics[0] = recall[0]

        if (label_0_present and label_1_present):
            metrics[1] = recall[1]
        elif (label_1_present):
            metrics[1] = recall[0]

        if (label_0_predicted):
            metrics[2] = prec_score[0]

        if (label_0_predicted and label_1_predicted):
            metrics[3] = prec_score[1]
        elif (label_1_predicted):
            metrics[3] = prec_score[0]

        return float(metrics[0]), float(metrics[1]), float(metrics[2]), float(metrics[3])                       

    def do_cv(self):
        #self.__file = open(self.__debug_file_name, "w")
        #self.__cv_file = open(self.__cv_file_name, "w")

        self.__db.read_db()

        #if (self.__mode == 1):
        #    max_recall_0_mean, estimator = self.cv_per_trace()
        #else:
        max_recall_0_mean = self.cv_per_problem()
        
        #self.__file.close()
        #self.__cv_file.close()

        return max_recall_0_mean

    def loocv_per_problem(self, classifier, features_list, labels_list):
        #all_training_problems = self.__db.get_all_training_problems()
        scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', \
                    'precision_macro', 'recall_macro', 'f1_macro', \
                    'precision_weighted', 'recall_weighted', 'f1_weighted']
        #clf = svm.SVC(kernel='linear', C=0.5)
        #clf = LinearSVC(random_state=0, C=0.000001) 
        #multi_class='crammer_singer')
        #clf = svm.SVC(C=500, decision_function_shape='ovr')
        #clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
        #clf = RadiusNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
        #clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')
        #clf = LogisticRegression(C=1.0, solver='sag', multi_class='multinomial')
        #clf = LogisticRegression(solver='sag')
        #clf = LogisticRegression(solver='saga')
        #clf = LogisticRegression(solver='lbfgs')
        #clf = LogisticRegressionCV(solver='newton-cg')
        #clf = MLPClassifier(solver='sgd')
        #clf = DecisionTreeClassifier(random_state=0)
        #clf = RandomForestClassifier(max_depth=20, random_state=0)
        #clf = AdaBoostClassifier()
        #clf = GradientBoostingClassifier(max_depth=3)
        #clf = BaggingClassifier(n_estimators=20)

        #n_estimators = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        #criterion = ['gini', 'entropy']
        #max_depth = [None, 20, 30, 50]
        #min_samples_split = [2, 12, 22]
        #max_features = [None, 'log2', 'sqrt', 'auto']

        n_estimators = [80, 83, 85, 87, 90, 91, 93, 95, 97, 100, 102, 105, 107, 110]
        criterion = ['gini']
        max_depth = [None]
        min_samples_split = [2]
        max_features = ['auto']
        #loss = ['deviance']

        # ExtraTree
        self.__file.write("n_estimators, criterion, max_depth, min_samples_split, max_features, \
                        recall_score_0, recall_score_1, recall_score_2, precision_score_0, precision_score_1, precision_score_2, \
                        f1_score_0, f1_score_1, f1_score_2, recall_score_macro, recall_score_micro, precision_score_macro, \
                        precision_score_micro, f1_score_macro, f1_score_micro\n")
        #self.__file.write("n_estimators, loss, max_depth, min_samples_split, max_features, \
        #                recall_score_0, recall_score_1, recall_score_2, precision_score_0, precision_score_1, precision_score_2, \
        #                f1_score_0, f1_score_1, f1_score_2, recall_score_macro, recall_score_micro, precision_score_macro, \
        #                precision_score_micro, f1_score_macro, f1_score_micro\n")
        self.__cv_file.write("n_estimators, criterion, max_depth, min_samples_split, max_features, \
                        recall_0_avg, recall_0_dev, recall_1_avg, recall_1_dev, recall_2_avg, recall_2_dev,\
                        recall_macro, recall_micro, recall_weighted, precision_0_avg, precision_0_dev, \
                        precision_1_avg, precision_1_dev, precision_2_avg, precision_2_dev, \
                        precision_macro, precision_micro, precision_weighted, f1_0_avg, f1_0_dev, \
                        f1_1_avg, f1_1_dev, f1_2_avg, f1_2_dev, f1_macro, f1_micro, f1_weighted\n")

        #GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, \
                #subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, \
                #min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, \
                #min_impurity_split=None, init=None, random_state=None, max_features=None, \
                #verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’)
        max_recall_0_mean = 0
        best_recall_1_mean = 0
        best_recall_2_mean = 0
        best_estimator_count = 10 # Start with the default value

        for estimator in n_estimators:
            for c in criterion:
                for depth in max_depth:
                    for sample_split in min_samples_split:
                        for features in max_features:
                            recall_0 = []
                            recall_1 = []
                            recall_2 = []

                            precision_0 = []
                            precision_1 = []
                            precision_2 = []

                            f1_0 = []
                            f1_1 = []
                            f1_2 = []

                            clf = ExtraTreesClassifier(n_estimators=estimator, criterion=c, \
                                    max_depth=depth, min_samples_split=sample_split, \
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
                                    max_features=features, max_leaf_nodes=None, \
                                    min_impurity_decrease=0.0, min_impurity_split=None, \
                                    bootstrap=False, oob_score=False, \
                                    n_jobs=1, random_state=None, verbose=0, \
                                    warm_start=False, class_weight=None)
        
                            #clf = GradientBoostingClassifier(loss=l, learning_rate=0.1, n_estimators=estimator, \
                            #        subsample=1.0, criterion='friedman_mse', min_samples_split=sample_split, min_samples_leaf=1, \
                            #        min_weight_fraction_leaf=0.0, max_depth=depth, min_impurity_decrease=0.0, \
                            #        min_impurity_split=None, init=None, random_state=None, max_features=features, \
                            #        verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
                            skf = StratifiedKFold(n_splits=self.__fold)
                            skf.get_n_splits(features_list, labels_list)
                            #features_list = all_training_problems.get_features_list()
                            #labels_list = all_training_problems.get_label_list()
                            fold = 0
                            for train_index, test_index in skf.split(features_list,  labels_list):
                                #print("TRAIN:", train_index, "TEST:", test_index)
            
                                tr_features = list()
                                tr_labels = list()
                                tt_features = list()
                                tt_labels = list()

                                for i in train_index:
                                    tr_features.append(features_list[i])
                                    tr_labels.append(labels_list[i])

                                for i in test_index:
                                    tt_features.append(features_list[i])
                                    tt_labels.append(labels_list[i])

                                #print(tr_features)
                                clf.fit(tr_features, tr_labels)
                                predicted_labels = list()
                                train_predicted_labels = list()

                                #for problem in tr_features:
                                #    predicted_label = clf.predict([problem])
                                #    train_predicted_labels.append(predicted_label)

                                #i = 0
                                #while i < len(train_predicted_labels):
                                    #print("Predicted,  Actual ", train_predicted_labels[i], tr_labels[i])
                                #    i = i + 1

                                #print("Confusion Matrix for train data = ", confusion_matrix(tr_labels, train_predicted_labels))

                                for problem in tt_features:
                                    #print(problem)
                                    predicted_label = clf.predict([problem])
                                    predicted_labels.append(predicted_label)

                                i = 0
                                #   while i < len(predicted_labels):
                                #print("Predicted,  Actual ", predicted_labels[i], tt_labels[i])
                                #       i = i + 1

                                print("&&&&&&&&&& Fold = ", fold, "&&&&&&&&&&")
                                fold = fold + 1
                                print("Confusion Matrix = ", confusion_matrix(tt_labels, predicted_labels))
            
                                recall = recall_score(tt_labels, predicted_labels, average=None)
                                print("recall_score = ", recall)

                                prec_score = precision_score(tt_labels, predicted_labels, average=None)
                                print("precision_score = ", prec_score)
                               
                                f1 = f1_score(tt_labels, predicted_labels, average=None)
                                print("f1_score = ", f1)
                                   
                                recall_macro = recall_score(tt_labels, predicted_labels, average='macro')
                                print("recall_score_macro = ", recall_macro)
                                
                                recall_micro = recall_score(tt_labels, predicted_labels, average='micro')
                                print("recall_score_micro = ", recall_micro)
                                
                                precision_macro = precision_score(tt_labels, predicted_labels, average='macro')    
                                print("precision_score_macro = ", precision_macro)
                                    
                                precision_micro = precision_score(tt_labels, predicted_labels, average='micro')    
                                print("precision_score_micro = ", precision_micro)
                                   
                                f1_macro = f1_score(tt_labels, predicted_labels, average='macro') 
                                print("f1_score_macro = ", f1_macro)
                                    
                                f1_micro = f1_score(tt_labels, predicted_labels, average='micro')
                                print("f1_score_micro = ", f1_micro)
                                print("&&&&&&&&&&&&&&&&&&&&&&&&&")
                                recall_0.append(recall[0])
                                recall_1.append(recall[1])
                                recall_2.append(recall[2])

                                precision_0.append(prec_score[0])
                                precision_1.append(prec_score[1])
                                precision_2.append(prec_score[2])

                                f1_0.append(f1[0])
                                f1_1.append(f1[1])
                                f1_2.append(f1[2])
                                #row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                #    + str(sample_split) + ", " + str(features) + ", " \
                                #    + str(recall[0]) + ", " + str(recall[1])  + ", " + str(recall[2]) + ", " \
                                #    + str(prec_score[0]) + ", " + str(prec_score[1])  + ", " + str(prec_score[2]) + ", " \
                                #    + str(f1[0]) + ", " + str(f1[1])  + ", " + str(f1[2]) + ", " \
                                #    + str(recall_macro) + ", " + str(recall_micro) + ", " + str(precision_macro) + ", " \
                                #    + str(precision_micro) + ", " + str(f1_macro) + ", " + str(f1_micro) +  '\n'

                                row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                    + str(sample_split) + ", " + str(features) + ", " \
                                    + str(recall[0]) + ", " + str(recall[1])  + ", " + str(recall[2]) + ", " \
                                    + str(prec_score[0]) + ", " + str(prec_score[1])  + ", " + str(prec_score[2]) + ", " \
                                    + str(f1[0]) + ", " + str(f1[1])  + ", " + str(f1[2]) + ", " \
                                    + str(recall_macro) + ", " + str(recall_micro) + ", " + str(precision_macro) + ", " \
                                    + str(precision_micro) + ", " + str(f1_macro) + ", " + str(f1_micro) +  '\n'
                                self.__file.write(row)
                                #self.__cv_file.write(row)

                            scores = cross_validate(clf, features_list, labels_list, scoring=scoring, cv=skf, return_train_score=True)
                            #print("score keys = ", sorted(scores.keys()))
                            #print("f1_score_micro = ", scores['test_f1_micro'])
                            print(str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                    + str(sample_split) + ", " + str(features))
                            print("*******************************************************")
                            recall_0_mean = mean(recall_0)
                            print("recall_0_mean = ", recall_0_mean)
                            recall_0_dev = stdev(recall_0)
                            print("recall_0_dev = ", recall_0_dev)
                            
                            recall_1_mean = mean(recall_1)
                            print("recall_1_mean = ", recall_1_mean)
                            recall_1_dev = stdev(recall_1)
                            print("recall_1_dev = ", recall_1_dev)
                            
                            recall_2_mean = mean(recall_2)
                            print("recall_2_mean = ", recall_2_mean)
                            recall_2_dev = stdev(recall_2)
                            print("recall_2_dev = ", recall_2_dev)
                            
                            #print("recall_macro = ", scores['test_recall_macro'])
                            recall_macro_mean = scores['test_recall_macro'].mean()
                            print("recall_macro_mean = ", recall_macro_mean)
                            #print("recall_macro_std = ", scores['test_recall_macro'].std())

                            #print("recall_micro = ", scores['test_recall_micro'])
                            recall_micro_mean = scores['test_recall_micro'].mean()
                            print("recall_micro_mean = ", recall_micro_mean)
                            #print("recall_micro_std = ", scores['test_recall_micro'].std())
        
                            #print("recall_weighted = ", scores['test_recall_weighted'])
                            recall_weighted_mean = scores['test_recall_weighted'].mean()
                            print("recall_weighted_mean = ", recall_weighted_mean)
                            #print("recall_weighted_std = ", scores['test_recall_weighted'].std())
                            
                            print("#########################")
                            precision_0_mean = mean(precision_0)
                            print("precision_0_mean = ", precision_0_mean)
                            precision_0_dev = stdev(precision_0)
                            print("precision_0_dev = ", precision_0_dev)
                            
                            precision_1_mean = mean(precision_1)
                            print("precision_1_mean = ", precision_1_mean)
                            precision_1_dev = stdev(precision_1)
                            print("precision_1_dev = ", precision_1_dev)
                            
                            precision_2_mean = mean(precision_2)
                            print("precision_2_mean = ", precision_2_mean)
                            precision_2_dev = stdev(precision_2)
                            print("precision_2_dev = ", precision_2_dev)
                            
                            #print("precision_macro = ", scores['test_precision_macro'].mean())
                            precision_macro_mean = scores['test_precision_macro'].mean()
                            print("precision_macro_mean = ", precision_macro_mean)
                            #print("precision_macro_std = ", scores['test_precision_macro'].std())
        
                            #print("precision_micro = ", scores['test_precision_micro'].mean())
                            precision_micro_mean = scores['test_precision_micro'].mean()
                            print("precision_micro_mean = ", precision_micro_mean)
                            #print("precision_micro_std = ", scores['test_precision_micro'].std())
        
                            #print("precision_weighted = ", scores['test_precision_weighted'])
                            precision_weighted_mean = scores['test_precision_weighted'].mean()
                            print("precision_weighted_mean = ", precision_weighted_mean)
                            #print("precision_weighted_std = ", scores['test_precision_weighted'].std())
                            print("#########################")
                            f1_0_mean = mean(f1_0)
                            print("f1_0_mean = ", f1_0_mean)
                            f1_0_dev = stdev(f1_0)
                            print("f1_0_dev = ", f1_0_dev)
                            
                            f1_1_mean = mean(f1_1)
                            print("f1_1_mean = ", f1_1_mean)
                            f1_1_dev = stdev(f1_1)
                            print("f1_1_dev = ", f1_1_dev)
                            
                            f1_2_mean = mean(f1_2)
                            print("f1_2_mean = ", f1_2_mean)
                            f1_2_dev = stdev(f1_2)
                            print("f1_2_dev = ", f1_2_dev)
                            
                            f1_score_macro_mean = (scores['test_f1_macro']).mean()
                            print("f1_score_macro mean = ", f1_score_macro_mean)
                            #print("f1_score_macro std = ", (scores['test_f1_macro']).std())
        
                            f1_score_micro_mean = (scores['test_f1_micro']).mean()
                            print("f1_score_micro mean = ", f1_score_micro_mean)
                            #print("f1_score_micro std = ", (scores['test_f1_micro']).std())
        
                            f1_score_weighted_mean = scores['test_f1_weighted'].mean()
                            print("f1_score_weighted_mean = ", f1_score_weighted_mean)
                            #print("f1_score_weighted_std = ", scores['test_f1_weighted'].std())
                            print("*******************************************************")

                            if (max_recall_0_mean < recall_0_mean):
                                max_recall_0_mean = recall_0_mean
                                best_recall_1_mean = recall_1_mean
                                best_recall_2_mean = recall_2_mean
                                best_estimator_count = estimator

                            cv_row = str(estimator) + ", " + str(c) + ", " + str(depth) + ", " \
                                    + str(sample_split) + ", " + str(features) + ", " \
                                    + str(recall_0_mean) + ", " + str(recall_0_dev) + ", " \
                                    + str(recall_1_mean) + ", " + str(recall_1_dev) + ", " \
                                    + str(recall_2_mean) + ", " + str(recall_2_dev) + ", " \
                                    + str(recall_macro_mean) + ", " + str(recall_micro_mean) + ", " \
                                    + str(recall_weighted_mean) + ", " \
                                    + str(precision_0_mean) + ", " + str(precision_0_dev) + ", " \
                                    + str(precision_1_mean) + ", " + str(precision_1_dev) + ", " \
                                    + str(precision_2_mean) + ", " + str(precision_2_dev) + ", " \
                                    + str(precision_macro_mean) + ", " \
                                    + str(precision_micro_mean) + ", " + str(precision_weighted_mean) + ", " \
                                    + str(f1_0_mean) + ", " + str(f1_0_dev) + ", " \
                                    + str(f1_1_mean) + ", " + str(f1_1_dev) + ", " \
                                    + str(f1_2_mean) + ", " + str(f1_2_dev) + ", " \
                                    + str(f1_score_macro_mean) + ", " + str(f1_score_micro_mean) + ", " \
                                    + str(f1_score_weighted_mean) + '\n'
                            
                            self.__cv_file.write(cv_row)
        #print("Score", scores)

        #print("f1_score_samples = ", scores['test_f1_samples'])
        #print("precision_samples = ", scores['test_precision_samples'])
        #print("recall_samples = ", scores['test_recall_samples'])
        #print("accuracy = ", scores['test_accuracy'])

        return max_recall_0_mean, best_recall_1_mean, best_recall_2_mean, best_estimator_count

    def do_loocv(self, classifier):
        self.__db.read_db()
        train_problems = self.__db.get_not_ignored_seed_problems()
        #test_problems = self.__db.get_ignored_seed_problems()
        features_list = StandardScaler().fit_transform(train_problems.get_features_list())
        X_resampled, y_resampled = ADASYN().fit_resample(features_list, \
                train_problems.get_label_list())
        self.__features_list = X_resampled
        self.__labels_list = y_resampled
        max_recall_0_mean, recall_1_mean, recall_2_mean, estimators \
                = self.loocv_per_problem(classifier, X_resampled, y_resampled)
        #max_recall_0_mean = 0
        #recall_1_mean = 0
        #recall_2_mean = 0
        #estimators = 80

        return max_recall_0_mean, recall_1_mean, recall_2_mean, estimators

    def get_features_list(self):
        return self.__features_list

def test_classifier(X_resampled, y_resampled, all_test_problems, classifier):
    print("##### Classifier = ", classifier, "#####")
    #X_resampled, y_resampled = SMOTE().fit_resample(all_training_problems.get_features_list(), \
    #            all_training_problems.get_label_list())
    #X_resampled, y_resampled = ADASYN().fit_resample(all_training_problems.get_features_list(), \
    #            all_training_problems.get_label_list())
    #X_resampled, y_resampled = BorderlineSMOTE().fit_resample(all_training_problems.get_features_list(), \
    #            all_training_problems.get_label_list())
    #smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
    #X_resampled, y_resampled = smote_nc.fit_resample(all_training_problems.get_features_list(), \
    #            all_training_problems.get_label_list())
    #clf = svm.SVC(kernel='linear', C=0.5)
    #clf = LinearSVC(random_state=0, C=0.000001) 
    #multi_class='crammer_singer')
    #clf = svm.SVC(C=500, decision_function_shape='ovr')
    if (classifier == "KNeighborsClassifier"):
        clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
    if (classifier == "RadiusNeighborsClassifier"):
        clf = RadiusNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree')
    if (classifier == "MLPClassifier"):
        clf = MLPClassifier(solver='sgd') # precision/recall for label 0 is always 0
    if (classifier == "DecisionTreeClassifier"):
        clf = DecisionTreeClassifier(random_state=None, max_depth=100, max_features=None, class_weight='balanced')
    if (classifier == "DecisionTreeClassifierWeight"):
        clf = DecisionTreeClassifier(random_state=0, max_depth=100,max_features=None, class_weight= {0:0.4, 1:.2, 2:0.2})
    if (classifier == "RandomForestClassifier"):
        clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
    if (classifier == "ExtraTreesClassifier"):
        clf = ExtraTreesClassifier(n_estimators=20)
    if (classifier == "AdaBoostClassifier"):
        clf = AdaBoostClassifier()
    if (classifier == "GradientBoostingClassifier"):
        clf = GradientBoostingClassifier(n_estimators=20, max_depth=5) # Reasonbale precision for 0, recall/precision for 2
    if (classifier == "BaggingClassifier"):
        clf = BaggingClassifier(random_state=0, n_estimators=250)

    #clf = LogisticRegression(C=1.0, solver='newton-cg', multi_class='multinomial')
    #clf = LogisticRegression(C=1.0, solver='sag', multi_class='multinomial')
    #clf = LogisticRegression(solver='sag')
    #clf = LogisticRegression(solver='saga')
    #clf = LogisticRegression(solver='lbfgs')
    #clf = LogisticRegressionCV(solver='newton-cg')
    # Good one
    #clf = DecisionTreeClassifier(random_state=0, max_depth=100,max_features=None, class_weight= {0:0.4, 1:.2, 2:0.2})
    #clf = BaggingClassifier(random_state=0, n_estimators=20) # Good one
    #clf = IsolationForest()
    #clf = VotingClassifier(estimators=100)
    #class_weight = dict({0:3, 1:1, 2:2})
    #clf = RandomForestClassifier(bootstrap=True, \
    #        class_weight=class_weight, \
    #        criterion='gini', \
    #        max_depth=8, max_features='auto', max_leaf_nodes=None, \
    #        min_impurity_decrease=0.0, min_impurity_split=None, \
    #        min_samples_leaf=4, min_samples_split=10, \
    #        min_weight_fraction_leaf=0.0, n_estimators=20, \
    #        oob_score=False, \
    #        verbose=0, warm_start=False)
            #random_state=1, \
    clf.fit(X_resampled, y_resampled)
    predicted_labels = list()

    #all_test_problems = test_db.get_all_training_problems()
    problems = all_test_problems.get_features_list()

    for problem in problems:
         predicted_label = clf.predict([problem])
         predicted_labels.append(predicted_label)
   
    merge_labels = False
    problem_labels = all_test_problems.get_label_list()

    if (merge_labels):
        print("%%%%%%%%%%%% Merging Labels %%%%%%%%%%")
        i = 0
        while i < len(predicted_labels):
            if (predicted_labels[i] == 2):
                #print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                predicted_labels[i] = 1
                #print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])

            if (problem_labels[i] == 2):
                #print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                problem_labels[i] = 1
                #print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])
            
                #if (problem_labels[i] == 0 and predicted_labels[i] == 2):
                #    print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                #    predicted_labels[i] =[0]
                #    print("## Actual, Predicted ", problem_labels[i], predicted_labels[i])

                #if (problem_labels[i] == 1 and predicted_labels[i] == 2):
                #    print("Actual, Predicted ", problem_labels[i], predicted_labels[i])
                #    predicted_labels[i] = [0]
                #    print("### Actual, Predicted ", problem_labels[i], predicted_labels[i])
            i = i + 1
 
    print("Confusion Matrix = ", confusion_matrix(problem_labels, predicted_labels))

    recall = recall_score(all_test_problems.get_label_list(), predicted_labels, average=None)
    print("recall_score = ", recall)

    prec_score = precision_score(all_test_problems.get_label_list(), predicted_labels, average=None)
    print("precision_score = ", prec_score)
                               
    f1 = f1_score(all_test_problems.get_label_list(), predicted_labels, average=None)
    print("f1_score = ", f1)
                                   
    recall_macro = recall_score(all_test_problems.get_label_list(), predicted_labels, average='macro')
    print("recall_score_macro = ", recall_macro)
                                
    recall_micro = recall_score(all_test_problems.get_label_list(), predicted_labels, average='micro')
    print("recall_score_micro = ", recall_micro)
                                
    precision_macro = precision_score(all_test_problems.get_label_list(), predicted_labels, average='macro')    
    print("precision_score_macro = ", precision_macro)
                                    
    precision_micro = precision_score(all_test_problems.get_label_list(), predicted_labels, average='micro')    
    print("precision_score_micro = ", precision_micro)
                                   
    f1_macro = f1_score(all_test_problems.get_label_list(), predicted_labels, average='macro') 
    print("f1_score_macro = ", f1_macro)
                                    
    f1_micro = f1_score(all_test_problems.get_label_list(), predicted_labels, average='micro')
    print("f1_score_micro = ", f1_micro)

def test(train_data_file, test_data_file):
    train_db = DBWrapper(train_data_file)
    train_db.read_db()
    all_training_problems = train_db.get_all_training_problems()

    test_db = DBWrapper(test_data_file)
    test_db.read_db()
    
    #X_resampled, y_resampled = SMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())
    X_resampled, y_resampled = ADASYN().fit_resample(train_problems.get_features_list(), \
                train_problems.get_label_list())
    #X_resampled, y_resampled = BorderlineSMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())

    test_classifier(X_resampled, y_resampled, test_db.get_all_training_problems(), "ExtraTreesClassifier")
    

def test_mode_2(data_file, test_seeds):
    problem_db = DBWrapper(data_file, test_seeds)
    problem_db.read_db()
    #train_problems = problem_db.get_not_ignored_seed_problems()
    train_problems = problem_db.get_all_training_problems()
    test_problems = problem_db.get_ignored_seed_problems()
    print("train problem count = ", train_problems.get_db_size())
    print("Label 0 count = ", test_problems.get_label_list().count(0))
    print("Label 1 count = ", test_problems.get_label_list().count(1))
    print("Label 2 count = ", test_problems.get_label_list().count(2))

    X_resampled = train_problems.get_features_list()
    y_resampled = train_problems.get_label_list()

    print("****** Default *******")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")
    return
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")

    return
    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = SMOTE().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = SMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())

    print("****** SMOTE *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")
    
    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = ADASYN().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = ADASYN().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())
    
    print("****** ADASYN *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")
    
    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = BorderlineSMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())

    print("****** BorderlineSMOTE *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")

def test_mode_3(data_file, test_seeds):
    problem_db = DBWrapper(data_file, test_seeds)
    problem_db.read_db()
    train_problems = problem_db.get_not_ignored_seed_problems()
    test_problems = problem_db.get_ignored_seed_problems()

    print("Label 0 count = ", test_problems.get_label_list().count(0))
    print("Label 1 count = ", test_problems.get_label_list().count(1))
    print("Label 2 count = ", test_problems.get_label_list().count(2))

    total_features = train_problems.get_features_list() + test_problems.get_features_list()
    print("total_features = ", len(total_features))
    total_labels = train_problems.get_label_list() + test_problems.get_label_list()
    print("total_labels = ", len(total_labels))

    #features_list = StandardScaler().fit_transform(total_features)
    features_list = total_features
    X_resampled, y_resampled = SMOTE().fit_resample(features_list, \
                total_labels)

    assert(len(X_resampled) == len(y_resampled))

    print("total X_resampled = ", len(X_resampled))
    print("total Y_resampled = ", len(y_resampled))

    #s_x = set(X_resampled)
    #s_y = set(Y_resampled)

    #print("total s_x = ", len(s_x))
    #print("total s_y = ", len(s_y))
    
    final_train_features = list()
    final_train_labels = list()

    test_features = test_problems.get_features_list()
    test_labels = test_problems.get_label_list()

    i = 0

    a = [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   2.23891000e-02,   2.23891000e-02,   8.77379000e-02, \
        8.77379000e-02,   2.86217000e-02,   2.86217000e-02,   7.62440000e-02, \
        7.62440000e-02,   2.98783000e-01,   2.98783000e-01,   9.74684000e-02, \
        9.74684000e-02,   2.23891000e-02,   2.23891000e-02,   8.77379000e-02, \
        8.77379000e-02,   2.86217000e-02,   2.86217000e-02,  2.56395000e-02, \
        2.56395000e-02,   9.97498000e-02,   9.97498000e-02,   3.18571000e-02, \
        3.18571000e-02,   8.73130000e-02,   8.73130000e-02,   3.39688000e-01, \
        3.39688000e-01,   1.08486000e-01,   1.08486000e-01,   2.56395000e-02, \
        2.56395000e-02,   9.97498000e-02,   9.97498000e-02,   3.18571000e-02, \
        3.18571000e-02,   2.38330000e-02,   2.38330000e-02,   9.82116000e-02, \
        9.82116000e-02,   3.21142000e-02,   3.21142000e-02,   8.11610000e-02, \
        8.11610000e-02,   3.34450000e-01,   3.34450000e-01,   1.09362000e-01, \
        1.09362000e-01,   2.38330000e-02,   2.38330000e-02,   9.82116000e-02, \
        9.82116000e-02,   3.21142000e-02,   3.21142000e-02,   1.59221000e-04, \
        1.59221000e-04,   7.11760000e-03,   7.11760000e-03,   8.15938000e-03, \
        8.15938000e-03,   5.42211000e-04,   5.42211000e-04,   2.42383000e-02, \
        2.42383000e-02,   2.77860000e-02,   2.77860000e-02,   1.59221000e-04, \
        1.59221000e-04,   7.11760000e-03,   7.11760000e-03,   8.15938000e-03, \
        8.15938000e-03,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,   0.00000000e+00, \
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00]

    print(a in test_features)

    while (i < len(X_resampled)):
        #print("X_resampled[i]", X_resampled[i])
        
        j = 0;
        f = list()
        while (j < len(X_resampled[i])):
            f.append(X_resampled[i][j])
            j = j+1

        #print("f = ", f)

        if (not (f in test_features)):
        #if (not test_features.any(X_resampled[i])):
            final_train_features.append(X_resampled[i])
            final_train_labels.append(y_resampled[i])

        i = i + 1

    print("total final_train_features = ", len(final_train_features))
    print("total final_train_labels = ", len(final_train_labels))
    #s_test_x = set(test_problems.get_features_list())
    #s_test_y = set(test_problems.get_label_list())

    #s_train_x = s_x - 
    return

    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = SMOTE().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = SMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())

    print("****** SMOTE *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")
    
    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = ADASYN().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = ADASYN().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())
    
    print("****** ADASYN *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")
    
    features_list = StandardScaler().fit_transform(train_problems.get_features_list())
    X_resampled, y_resampled = BorderlineSMOTE().fit_resample(features_list, \
                train_problems.get_label_list())
    #X_resampled, y_resampled = BorderlineSMOTE().fit_resample(train_problems.get_features_list(), \
    #            train_problems.get_label_list())

    print("****** BorderlineSMOTE *******")
    test_classifier(X_resampled, y_resampled, test_problems, "KNeighborsClassifier")
    #test_classifier(X_resampled, y_resampled, test_problems, "RadiusNeighborsClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "MLPClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "DecisionTreeClassifierWeight")
    test_classifier(X_resampled, y_resampled, test_problems, "RandomForestClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "ExtraTreesClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "AdaBoostClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "GradientBoostingClassifier")
    test_classifier(X_resampled, y_resampled, test_problems, "BaggingClassifier")


#def test(data_file, fold, iterations, ignored_seeds, merge_labels = True):
#    cv = CrossValidate(data_file, fold, iterations, ignored_seed)
#    cv.read_file()
#    return cv.test_ignored_traces(merge_labels)

# First argument - input csv file
# Second argument - 1/0 - CV based on trace/problem
# Third argument - skip trace

def main():
    # print command line arguments
    #for arg in sys.argv[1:]:
    #    print (arg)

    #print (sys.argv[0])
    print (sys.argv[1])
    #print (sys.argv[3])
    test_mode = 2
    merge_labels = False

    #if (len(sys.argv) > 5):
    #    merge_labels = int(sys.argv[5])

    #print("Merge_labels = ", merge_labels)
    #print("Parsing to get seeds")
    #db_wrapper = DBWrapper(sys.argv[1])
    #db_wrapper.read_db()
    #all_seeds = db_wrapper.get_all_seeds()
    #print("all_seeds = ", all_seeds)
    #seed_list = list()

    #for seed in all_seeds:
    #    seed_list.append(seed)

    if (test_mode == 1):
        print (sys.argv[2])
        test(sys.argv[1], sys.argv[2])
    elif (test_mode == 2):
        # Random 70 seeds to test and remaining seeds to train.
        i = 0
        iterations = 1
        while (i < iterations):
            i = i + 1
            print ("iteration ##### ", i)
            #random_seeds = list(get_n_random_numbers(1, 700, 70))
            random_seeds = []
            random_seeds = [384, 640, 2, 257, 646, 392, 8, 522, 267, \
                    268, 526, 399, 16, 529, 658, 19, 147, 405, 659, 536, \
                    154, 412, 286, 160, 418, 162, 420, 294, 44, 178, 692, \
                    693, 53, 696, 572, 316, 60, 320, 324, 71, 201, 204, 593, \
                    466, 211, 473, 219, 220, 477, 224, 484, 357, 356, 103, 360, \
                    105, 361, 493, 494, 251, 240, 623, 242, 372, 505, 506, \
                    635, 253, 638, 639]
        #random_seeds = [13, 15, 16, 41, 51, 55, 67, 74, 76, 93, \
        #                98, 99, 108, 112, 120, 121, 127, 151, 155, \
        #                157, 172, 215, 216, 224, 255, 265, 280, 284, \
        #                289, 290, 294, 295, 304, 310, 321, 326, 332, \
        #                339, 346, 347, 403, 407, 416, 419, 424, 434, \
        #                442, 461, 483, 487, 490, 502, 526, 527, 529, \
        #                551, 553, 561, 584, 602, 615, 620, 627, 629, \
        #                638, 651, 655, 680, 694]

        #[640, 384, 2, 259, 514, 132, 518, 647, 264, 393, 650, 394, 140, 141, 397, 525, 13, 657, 274, 396, 532, 277, 534, 23, 536, 281, 408, 278, 30, 673, 41, 170, 169, 300, 429, 687, 688, 137, 182, 56, 441, 63, 320, 577, 196, 395, 74, 206, 207, 465, 219, 477, 607, 480, 98, 357, 232, 489, 104, 107, 492, 111, 624, 369, 370, 627, 371, 373, 502, 628]
        #Label 0 count =  39
        #Label 1 count =  831
        #Label 2 count =  429
        #random_seeds.sort()
#SMOTE, bagging
#[384, 640, 2, 257, 646, 392, 8, 522, 267, 268, 526, 399, 16, 529, 658, 19, 147, 405, 659, 536, 154, 412, 286, 160, 418, 162, 420, 294, 44, 178, 692, 693, 53, 696, 572, 316, 60, 320, 324, 71, 201, 204, 593, 466, 211, 473, 219, 220, 477, 224, 484, 357, 356, 103, 360, 105, 361, 493, 494, 251, 240, 623, 242, 372, 505, 506, 635, 253, 638, 639]
        
            #print("test_seeds = ", len(random_seeds))
            test_mode_2(sys.argv[1], random_seeds)
    elif (test_mode == 3):
        # Generates synthetic sample from all the given seeds than leave out 70 random seeds for test
        i = 0
        iterations = 1
        while (i < iterations):
            i = i + 1
            print ("iteration ##### ", i)
            #random_seeds = list(get_n_random_numbers(1, 700, 70))
            random_seeds = []
            random_seeds = [384, 640, 2, 257, 646, 392, 8, 522, 267, \
                    268, 526, 399, 16, 529, 658, 19, 147, 405, 659, 536, \
                    154, 412, 286, 160, 418, 162, 420, 294, 44, 178, 692, \
                    693, 53, 696, 572, 316, 60, 320, 324, 71, 201, 204, 593, \
                    466, 211, 473, 219, 220, 477, 224, 484, 357, 356, 103, 360, \
                    105, 361, 493, 494, 251, 240, 623, 242, 372, 505, 506, \
                    635, 253, 638, 639]
            test_mode_3(sys.argv[1], random_seeds)
    elif (test_mode == 4):
        # Leave one seed out of 70 seeds. Do cv on remaining 699 seeds
        random_seeds = []
        random_seeds = [384, 640, 2, 257, 646, 392, 8, 522, 267, \
                    268, 526, 399, 16, 529, 658, 19, 147, 405, 659, 536, \
                    154, 412, 286, 160, 418, 162, 420, 294, 44, 178, 692, \
                    693, 53, 696, 572, 316, 60, 320, 324, 71, 201, 204, 593, \
                    466, 211, 473, 219, 220, 477, 224, 484, 357, 356, 103, 360, \
                    105, 361, 493, 494, 251, 240, 623, 242, 372, 505, 506, \
                    635, 253, 638, 639]
        seed_debug_file_name = sys.argv[2] + "_seed_log.csv"
        debug_seed_file = open(seed_debug_file_name, "w")
        debug_seed_file.write("Seed, estimators, recall_score_0, recall_score_1, recall_score_2, \
                                recall_score, recall_macro, recall_micro, \
                                precision_score_0, precision_score_1, precision_score_2, \
                                precision_score, precision_macro, precision_micro, \
                                max_recall_0_mean, recall_1_mean, recall_2_mean \n")
        classifier = "ExtraTreesClassifier"
        all_seeds = []
        all_seeds.extend(range(1,701))

        for random_seed in random_seeds:
            debug_file_name = sys.argv[2] + "_" + str(random_seed) + "_" + classifier + "_log.csv"
            cv_file_name = sys.argv[2] + "_" + str(random_seed) + "_cv_" + classifier + "_log.csv"
            debug_file = open(debug_file_name, "w")
            cv_file = open(cv_file_name, "w")

            ignored_seeds = [random_seed]
            cv = CrossValidate(sys.argv[1], sys.argv[2], ignored_seeds, debug_file, cv_file)
            max_recall_0_mean, recall_1_mean, recall_2_mean, estimators = cv.do_loocv(classifier)
            print("max_recall_0_mean, estimators", max_recall_0_mean, estimators)
            recall_0, recall_1, prec_score_0, prec_score_1 = cv.test_ignored_seeds(merge_labels, estimators)
            row = str(random_seed) + "," + str(estimators) + "," + str(recall_0) + ", " + str(recall_1) + ","\
                  + str(prec_score_0) + ", " + str(prec_score_1) + "," + str(max_recall_0_mean) + ", " \
                  + str(recall_1_mean) + ", " + str(recall_2_mean) + '\n'
            debug_seed_file.write(row)
            print("recall_0, recall_1, prec_score_0, prec_score_1", recall_0, recall_1, prec_score_0, prec_score_1)
            debug_file.close()
            cv_file.close()
            break

        debug_seed_file.close()
    elif (test_mode == 5):
        # Do CV on 630 seeds. And evaluate on 70 seeds as sanity check
        print("ERROR: Mode 5 not implemented")
    else:
        print (sys.argv[2])
        iteration = 0
        recall_0_avg = 0
        recall_1_avg = 0
        precision_0_avg = 0
        precision_1_avg = 0

        while (iteration < int(sys.argv[3])):
            cv = CrossValidate(sys.argv[1], sys.argv[2], "", "")
            max_recall_0_mean = cv.do_cv()
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            
            iteration = iteration + 1

if __name__ == "__main__":
    sys.exit(main())
