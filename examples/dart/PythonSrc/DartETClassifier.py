import sys
from timeit import default_timer as timer
import time
#from ProblemDatabase import ProblemFeatures
#from ProblemDatabase import ProblemDatabase
#from ProblemDatabase import ProblemInstance
#from ProblemDatabase import DBWrapper
from DartDBWrapper import DBWrapper
from sklearn.ensemble import ExtraTreesClassifier

print("Sourcing file")

class ETClassifier:
    __slots__ = ['__data_file', '__skip_trace', '__db', '__clf', '__compare_past_workload_for_similarity']
    
    def __init__(self, data_file, skip_trace, estimators_count = 85, \
                    compare_past_workload_for_similarity = 0):
        print("Ignoring Trace while training = ", skip_trace)
        print("Estimators Count = ", estimators_count)
        #print("Compare past workload = ", compare_past_workload_for_similarity)
        self.__data_file = data_file
        self.__skip_trace = skip_trace
        self.__db = DBWrapper(data_file, [skip_trace])
        self.__clf = ExtraTreesClassifier(n_estimators=estimators_count)
        #self.__compare_past_workload_for_similarity = compare_past_workload_for_similarity
        
    def train(self):
        print("Inside train")
        self.__db.read_db()
        all_training_problems = self.__db.get_all_training_problems()
        self.__clf.fit(all_training_problems.get_features_list(), all_training_problems.get_label_list())
        print("Exiting Train")
        return 1

    def predict(self, features):
        print("Inside predict", features)
        t1 = timer()
        prediction = self.__clf.predict(features)
        #print("Prediction time in seconds = ", timer() - t1)

        return prediction

    #def fun(self):
        #print("Inside fun")
        #return 1

    #def get_request_arrival_rate(self, rar):
        #print("rar = ", rar)
    #    fixed_rar = float(rar)
        #print("fixed_rar = ", fixed_rar)
    
    #    if (fixed_rar == float('inf')):
    #        fixed_rar = sys.maxsize #sys.long_info.max #2147483647
            #sys.float_info.max
    #        print("Changed fixed_rar = ", fixed_rar)

    #    return fixed_rar

    def classifier_predict(self, altitude, formation, ecm, incAlt_state, \
                decAlt_state, incAlt2_state, decAlt2_state, \
                tr1, tr2, tr3, tr4, tr5, tr6, tr7, tr8, tr9, tr10, tr11, \
                tr12, tr13, tr14, tr15, tr16, tr17, tr18, tr19, tr20, tr21, \
                tr22, tr23, tr24, tr25, tr26, tr27, tr28, tr29, tr30, tr31, \
                tr32, tr33, tr34, tr35, tr36, tr37, tr38, tr39, tr40, tr41, \
                tr42, tr43, tr44, tr45, tr46, \
                th1, th2, th3, th4, th5, th6, th7, th8, th9, th10, th11, \
                th12, th13, th14, th15, th16, th17, th18, th19, th20, th21, \
                th22, th23, th24, th25, th26, th27, th28, th29, th30, th31, \
                th32, th33, th34, th35, th36, th37, th38, th39, th40, th41, \
                th42, th43, th44, th45, th46):

        #print("Inside classifier_predict", dimmer)
        features = list()
        
        features.append(int(altitude))
        features.append(int(formation)) 
        features.append(int(ecm)) 
        features.append(int(incAlt_state)) 
        features.append(int(decAlt_state)) 
        features.append(int(incAlt2_state)) 
        features.append(int(decAlt2_state))

        #if (self.__compare_past_workload_for_similarity == 1):
        features.append(float(tr1))
        features.append(float(th1))
        features.append(float(tr2))
        features.append(float(th2))
        features.append(float(tr3))
        features.append(float(th3))
        features.append(float(tr4))
        features.append(float(th4))
        features.append(float(tr5))
        features.append(float(th5))
        features.append(float(tr6))
        features.append(float(th6))
        features.append(float(tr7))
        features.append(float(th7))
        features.append(float(tr8))
        features.append(float(th8))
        features.append(float(tr9))
        features.append(float(th9))
        features.append(float(tr10))
        features.append(float(th10))
        features.append(float(tr11))
        features.append(float(th11))
        features.append(float(tr12))
        features.append(float(th12))
        features.append(float(tr13))
        features.append(float(th13))
        features.append(float(tr14))
        features.append(float(th14))
        features.append(float(tr15))
        features.append(float(th15))
        features.append(float(tr16))
        features.append(float(th16))
        features.append(float(tr17))
        features.append(float(th17))
        features.append(float(tr18))
        features.append(float(th18))
        features.append(float(tr19))
        features.append(float(th19))
        features.append(float(tr20))
        features.append(float(th20))
        features.append(float(tr21))
        features.append(float(th21))
        features.append(float(tr22))
        features.append(float(th22))
        features.append(float(tr23))
        features.append(float(th23))
        features.append(float(tr24))
        features.append(float(th24))
        features.append(float(tr25))
        features.append(float(th25))
        features.append(float(tr26))
        features.append(float(th26))
        features.append(float(tr27))
        features.append(float(th27))
        features.append(float(tr28))
        features.append(float(th28))
        features.append(float(tr29))
        features.append(float(th29))
        features.append(float(tr30))
        features.append(float(th30))
        features.append(float(tr31))
        features.append(float(th31))
        features.append(float(tr32))
        features.append(float(th32))
        features.append(float(tr33))
        features.append(float(th33))
        features.append(float(tr34))
        features.append(float(th34))
        features.append(float(tr35))
        features.append(float(th35))
        features.append(float(tr36))
        features.append(float(th36))
        features.append(float(tr37))
        features.append(float(th37))
        features.append(float(tr38))
        features.append(float(th38))
        features.append(float(tr39))
        features.append(float(th39))
        features.append(float(tr40))
        features.append(float(th40))
        features.append(float(tr41))
        features.append(float(th41))
        features.append(float(tr42))
        features.append(float(th42))
        features.append(float(tr43))
        features.append(float(th43))
        features.append(float(tr44))
        features.append(float(th44))
        features.append(float(tr45))
        features.append(float(th45))
        features.append(float(tr46))
        features.append(float(th46))
      
        prediction = self.predict([features])

        #print ("prediction = ", prediction[0])

        return int(prediction[0])
   
def main():
    classifier = None
    compare_past_workload_for_similarity = 0
    estimator_count = 87

    if (len(sys.argv) > 3):
        estimator_count = int(sys.argv[3])
    
    if (len(sys.argv) > 4):
        compare_past_workload_for_similarity = int(sys.argv[4])

    classifier = ETClassifier(sys.argv[1], sys.argv[2], estimator_count, \
                    compare_past_workload_for_similarity)

    classifier.train()
    #t1 = current_time()
    #start = time.time()
    #result = classifier.classifier_predict(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.008389, \
    #                12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, \
    #                119.205145, 234.066441, str('inf'), str('inf'), str('inf'), str('inf'))

    result = classifier.classifier_predict(2, 0, 0, 0, 0, 0, 0, 0,	0, 0.034225, \
            0.034225, 0.11655,	0.11655, 0.034225, 0.034225, 0.11655, 0.11655, \
            0.3969, 0.3969, 0.11655,	0.11655, 0.034225, 0.034225, 0.11655, \
            0.11655, 0.034225, 0.034225, 0.0289966, 0.0289966, 0.108627, 0.108627, \
            0.0334962, 0.0334962, 0.0987452, 0.0987452, 0.36992, 0.36992, 0.114068, \
            0.114068, 0.0289966, 0.0289966, 0.108627, 0.108627, 0.0334962, 0.0334962, \
            0, 0, 0, 0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0,	0, 0, 0, 0.0217548,	\
            0.0217548, 0.0993049, 0.0993049, 0.0330846,	0.0330846, 0.0740839, \
            0.0740839, 0.338173, 0.338173, 0.112667, 0.112667, 0.0217548, 0.0217548, \
            0.0993049, 0.0993049, 0.0330846,	0.0330846, 0, 0, 0,	0, 0, 0, 0,	0, 0, \
            0, 0, 0, 0, 0, 0, 0, 0, 0)

    #str(1), str(2), str(3), str(4), str(5), str(6), str(7),\
                                            #str(8), str(9), str(10), str(11), str(12), \
                                            #str(13), str(14), str(15), str(15), str(16), \
                                            #str(17), str(18), str(19), str(20), str(21), \
                                            #str(22), str(23), str(24), str(25), str(26), \
                                            #str(27), str(28), str(29), str(30))
    #result = classifier.predict([[1,2,3,4,5,6,7,8,9, 10, 11, float(12), float(13), \
    #                    float(14), float(15), float(16), float(17), float(18), float(19), \
    #                    float(20), float(21), float(22), float(23), \
    #                    float(24), float(25), float(132.222),  float(141.602),  \
    #                    float(147.459),  float(153.029),  float(158.291),  float(163.23)]])
    #end = time.time()
    #print(end - start)
    #print(current_time() - t1)
    print("Prediction =", result)
    return result
            
if __name__ == "__main__":
    sys.exit(main())
