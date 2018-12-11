from sklearn.neighbors import KNeighborsClassifier
import sys
from collections import defaultdict

# TODO this class needs to be merged in class ProblemDatabase
class ProblemFeatures:
    def __init__(self, altitude, formation, ecm, incAlt_state, decAlt_state, incAlt2_state, \
            decAlt2_state, target, threat):
        #print("Inside ProblemFestures::init")
        self.__altitude = int(altitude)
        self.__formation = int(formation)
        self.__ecm = int(ecm)
        self.__incAlt_state = int(incAlt_state)
        self.__decAlt_state = int(decAlt_state)
        self.__incAlt2_state = int(incAlt2_state)
        self.__decAlt2_state = int(decAlt2_state)
        
        #workload = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        self.__tr1 = float(target[0])
        self.__th1 = float(threat[0])
        self.__tr2 = float(target[1])
        self.__th2 = float(threat[1])
        self.__tr3 = float(target[2])
        self.__th3 = float(threat[2])
        self.__tr4 = float(target[3])
        self.__th4 = float(threat[3])
        self.__tr5 = float(target[4])
        self.__th5 = float(threat[4])
        self.__tr6 = float(target[5])
        self.__th6 = float(threat[5])
        self.__tr7 = float(target[6])
        self.__th7 = float(threat[6])
        self.__tr8 = float(target[7])
        self.__th8 = float(threat[7])
        self.__tr9 = float(target[8])
        self.__th9 = float(threat[8])
        self.__tr10 = float(target[9])
        self.__th10 = float(threat[9])
        self.__tr11 = float(target[10])
        self.__th11 = float(threat[10])
        self.__tr12 = float(target[11])
        self.__th12 = float(threat[11])
        self.__tr13 = float(target[12])
        self.__th13 = float(threat[12])
        self.__tr14 = float(target[13])
        self.__th14 = float(threat[13])
        self.__tr15 = float(target[14])
        self.__th15 = float(threat[14])
        self.__tr16 = float(target[15])
        self.__th16 = float(threat[15])
        self.__tr17 = float(target[16])
        self.__th17 = float(threat[16])
        self.__tr18 = float(target[17])
        self.__th18 = float(threat[17])
        self.__tr19 = float(target[18])
        self.__th19 = float(threat[18])
        self.__tr20 = float(target[19])
        self.__th20 = float(threat[19])
        self.__tr21 = float(target[20])
        self.__th21 = float(threat[20])
        self.__tr22 = float(target[21])
        self.__th22 = float(threat[21])
        self.__tr23 = float(target[22])
        self.__th23 = float(threat[22])
        self.__tr24 = float(target[23])
        self.__th24 = float(threat[23])
        self.__tr25 = float(target[24])
        self.__th25 = float(threat[24])
        self.__tr26 = float(target[25])
        self.__th26 = float(threat[25])
        self.__tr27 = float(target[26])
        self.__th27 = float(threat[26])
        self.__tr28 = float(target[27])
        self.__th28 = float(threat[27])
        self.__tr29 = float(target[28])
        self.__th29 = float(threat[28])
        self.__tr30 = float(target[29])
        self.__th30 = float(threat[29])
        self.__tr31 = float(target[30])
        self.__th31 = float(threat[30])
        self.__tr32 = float(target[31])
        self.__th32 = float(threat[31])
        self.__tr33 = float(target[32])
        self.__th33 = float(threat[32])
        self.__tr34 = float(target[33])
        self.__th34 = float(threat[33])
        self.__tr35 = float(target[34])
        self.__th35 = float(threat[34])
        self.__tr36 = float(target[35])
        self.__th36 = float(threat[35])
        self.__tr37 = float(target[36])
        self.__th37 = float(threat[36])
        self.__tr38 = float(target[37])
        self.__th38 = float(threat[37])
        self.__tr39 = float(target[38])
        self.__th39 = float(threat[38])
        self.__tr40 = float(target[39])
        self.__th40 = float(threat[39])
        self.__tr41 = float(target[40])
        self.__th41 = float(threat[40])
        self.__tr42 = float(target[41])
        self.__th42 = float(threat[41])
        self.__tr43 = float(target[42])
        self.__th43 = float(threat[42])
        self.__tr44 = float(target[43])
        self.__th44 = float(threat[43])
        self.__tr45 = float(target[44])
        self.__th45 = float(threat[44])
        self.__tr46 = float(target[45])
        self.__th46 = float(threat[45])
        #self.__r21 = workload[20]
        #self.__workload = workload

    #def find_similarity(self, problem):
    #    self.__serverA_count - problem.__serverA_count \
    #           + TimeSeriesSimilarity(self.__workload, problem.__workload)

    def print_features(self):
        print(self.__altitude)
        print(self.__formation)
        print(self.__ecm)
        print(self.__incAlt_state)
        print(self.__decAlt_state)
        print(self.__incAlt2_state)
        print(self.__decAlt2_state)

        #i = 0
        #while(i < 46):
        #    print(i, "  #target = ", self.__target[i])
        #    print(i, "  #threat = ", threat[i])
        #    i = i + 1

    def get_altitude(self):
        return self.__altitude

    def get_formation(self):
        return self.__formation

    def get_ecm(self):
        return self.__ecm

    def get_incAlt_state(self):
        return self.__incAlt_state

    def get_decAlt_state(self):
        return self.__decAlt_state

    def get_incAlt2_state(self):
        return self.__incAlt2_state

    def get_decAlt2_state(self):
        return self.__decAlt2_state

    def get_tr1(self):
        return self.__tr1

    def get_th1(self):
        return self.__th1

    def get_tr2(self):
        return self.__tr2

    def get_th2(self):
        return self.__th2

    def get_tr3(self):
        return self.__tr3

    def get_th3(self):
        return self.__th3

    def get_tr4(self):
        return self.__tr4

    def get_th4(self):
        return self.__th4

    def get_tr5(self):
        return self.__tr5

    def get_th5(self):
        return self.__th5

    def get_tr6(self):
        return self.__tr6

    def get_th6(self):
        return self.__th6

    def get_tr7(self):
        return self.__tr7

    def get_th7(self):
        return self.__th7

    def get_tr8(self):
        return self.__tr8

    def get_th8(self):
        return self.__th8

    def get_tr9(self):
        return self.__tr9

    def get_th9(self):
        return self.__th9

    def get_tr10(self):
        return self.__tr10

    def get_th10(self):
        return self.__th10

    def get_tr11(self):
        return self.__tr11

    def get_th11(self):
        return self.__th11

    def get_tr12(self):
        return self.__tr12

    def get_th12(self):
        return self.__th12

    def get_tr13(self):
        return self.__tr13

    def get_th13(self):
        return self.__th13

    def get_tr14(self):
        return self.__tr14

    def get_th14(self):
        return self.__th14

    def get_tr15(self):
        return self.__tr15

    def get_th15(self):
        return self.__th15

    def get_tr16(self):
        return self.__tr16

    def get_th16(self):
        return self.__th16

    def get_tr17(self):
        return self.__tr17

    def get_th17(self):
        return self.__th17

    def get_tr18(self):
        return self.__tr18

    def get_th18(self):
        return self.__th18

    def get_tr19(self):
        return self.__tr19

    def get_th19(self):
        return self.__th19

    def get_tr20(self):
        return self.__tr20

    def get_th20(self):
        return self.__th20

    def get_tr21(self):
        return self.__tr21

    def get_th21(self):
        return self.__th21

    def get_tr22(self):
        return self.__tr22

    def get_th22(self):
        return self.__th22

    def get_tr23(self):
        return self.__tr23

    def get_th23(self):
        return self.__th23

    def get_tr24(self):
        return self.__tr24

    def get_th24(self):
        return self.__th24

    def get_tr25(self):
        return self.__tr25

    def get_th25(self):
        return self.__th25

    def get_tr26(self):
        return self.__tr26

    def get_th26(self):
        return self.__th26

    def get_tr27(self):
        return self.__tr27

    def get_th27(self):
        return self.__th27

    def get_tr28(self):
        return self.__tr28

    def get_th28(self):
        return self.__th28

    def get_tr29(self):
        return self.__tr29

    def get_th29(self):
        return self.__th29

    def get_tr30(self):
        return self.__tr30

    def get_th30(self):
        return self.__th30

    def get_tr31(self):
        return self.__tr31

    def get_th31(self):
        return self.__th31

    def get_tr32(self):
        return self.__tr32

    def get_th32(self):
        return self.__th32

    def get_tr33(self):
        return self.__tr33

    def get_th33(self):
        return self.__th33

    def get_tr34(self):
        return self.__tr34

    def get_th34(self):
        return self.__th34

    def get_tr35(self):
        return self.__tr35

    def get_th35(self):
        return self.__th35

    def get_tr36(self):
        return self.__tr36

    def get_th36(self):
        return self.__th36

    def get_tr37(self):
        return self.__tr37

    def get_th37(self):
        return self.__th37

    def get_tr38(self):
        return self.__tr38

    def get_th38(self):
        return self.__th38

    def get_tr39(self):
        return self.__tr39

    def get_th39(self):
        return self.__th39

    def get_tr40(self):
        return self.__tr40

    def get_th40(self):
        return self.__th40

    def get_tr41(self):
        return self.__tr41

    def get_th41(self):
        return self.__th41

    def get_tr42(self):
        return self.__tr42

    def get_th42(self):
        return self.__th42

    def get_tr43(self):
        return self.__tr43

    def get_th43(self):
        return self.__th43

    def get_tr44(self):
        return self.__tr44

    def get_th44(self):
        return self.__th44

    def get_tr45(self):
        return self.__tr45

    def get_th45(self):
        return self.__th45

    def get_tr46(self):
        return self.__tr46

    def get_th46(self):
        return self.__th46

class ProblemInstance:
    __slots__ = ['__seed', '__profiling_dir', '__features', '__label', '__actual_label']

    def __init__(self, seed, profiling_dir, features, label, actual_label = -1):
        self.__seed = seed
        self.__profiling_dir = profiling_dir

        self.__features = list()

        self.__features.append(features.get_altitude())
        self.__features.append(features.get_formation())
        self.__features.append(features.get_ecm())
        self.__features.append(features.get_incAlt_state())
        self.__features.append(features.get_decAlt_state())
        self.__features.append(features.get_incAlt2_state())
        self.__features.append(features.get_decAlt2_state())
        self.__features.append(float(features.get_tr1()))
        self.__features.append(float(features.get_th1()))
        self.__features.append(float(features.get_tr2()))
        self.__features.append(float(features.get_th2()))
        self.__features.append(float(features.get_tr3()))
        self.__features.append(float(features.get_th3()))
        self.__features.append(float(features.get_tr4()))
        self.__features.append(float(features.get_th4()))
        self.__features.append(float(features.get_tr5()))
        self.__features.append(float(features.get_th5()))
        self.__features.append(float(features.get_tr6()))
        self.__features.append(float(features.get_th6()))
        self.__features.append(float(features.get_tr7()))
        self.__features.append(float(features.get_th7()))
        self.__features.append(float(features.get_tr8()))
        self.__features.append(float(features.get_th8()))
        self.__features.append(float(features.get_tr9()))
        self.__features.append(float(features.get_th9()))
        self.__features.append(float(features.get_tr10()))
        self.__features.append(float(features.get_th10()))
        self.__features.append(float(features.get_tr11()))
        self.__features.append(float(features.get_th11()))
        self.__features.append(float(features.get_tr12()))
        self.__features.append(float(features.get_th12()))
        self.__features.append(float(features.get_tr13()))
        self.__features.append(float(features.get_th13()))
        self.__features.append(float(features.get_tr14()))
        self.__features.append(float(features.get_th14()))
        self.__features.append(float(features.get_tr15()))
        self.__features.append(float(features.get_th15()))
        self.__features.append(float(features.get_tr16()))
        self.__features.append(float(features.get_th16()))
        self.__features.append(float(features.get_tr17()))
        self.__features.append(float(features.get_th17()))
        self.__features.append(float(features.get_tr18()))
        self.__features.append(float(features.get_th18()))
        self.__features.append(float(features.get_tr19()))
        self.__features.append(float(features.get_th19()))
        self.__features.append(float(features.get_tr20()))
        self.__features.append(float(features.get_th20()))
        self.__features.append(float(features.get_tr21()))
        self.__features.append(float(features.get_th21()))
        self.__features.append(float(features.get_tr22()))
        self.__features.append(float(features.get_th22()))
        self.__features.append(float(features.get_tr23()))
        self.__features.append(float(features.get_th23()))
        self.__features.append(float(features.get_tr24()))
        self.__features.append(float(features.get_th24()))
        self.__features.append(float(features.get_tr25()))
        self.__features.append(float(features.get_th25()))
        self.__features.append(float(features.get_tr26()))
        self.__features.append(float(features.get_th26()))
        self.__features.append(float(features.get_tr27()))
        self.__features.append(float(features.get_th27()))
        self.__features.append(float(features.get_tr28()))
        self.__features.append(float(features.get_th28()))
        self.__features.append(float(features.get_tr29()))
        self.__features.append(float(features.get_th29()))
        self.__features.append(float(features.get_tr30()))
        self.__features.append(float(features.get_th30()))
        self.__features.append(float(features.get_tr31()))
        self.__features.append(float(features.get_th31()))
        self.__features.append(float(features.get_tr32()))
        self.__features.append(float(features.get_th32()))
        self.__features.append(float(features.get_tr33()))
        self.__features.append(float(features.get_th33()))
        self.__features.append(float(features.get_tr34()))
        self.__features.append(float(features.get_th34()))
        self.__features.append(float(features.get_tr35()))
        self.__features.append(float(features.get_th35()))
        self.__features.append(float(features.get_tr36()))
        self.__features.append(float(features.get_th36()))
        self.__features.append(float(features.get_tr37()))
        self.__features.append(float(features.get_th37()))
        self.__features.append(float(features.get_tr38()))
        self.__features.append(float(features.get_th38()))
        self.__features.append(float(features.get_tr39()))
        self.__features.append(float(features.get_th39()))
        self.__features.append(float(features.get_tr40()))
        self.__features.append(float(features.get_th40()))
        self.__features.append(float(features.get_tr41()))
        self.__features.append(float(features.get_th41()))
        self.__features.append(float(features.get_tr42()))
        self.__features.append(float(features.get_th42()))
        self.__features.append(float(features.get_tr43()))
        self.__features.append(float(features.get_th43()))
        self.__features.append(float(features.get_tr44()))
        self.__features.append(float(features.get_th44()))
        self.__features.append(float(features.get_tr45()))
        self.__features.append(float(features.get_th45()))
        self.__features.append(float(features.get_tr46()))
        self.__features.append(float(features.get_th46()))
        
        #print(len(self.__features))
        
        self.__label = label
        self.__actual_label = actual_label

    def get_seed(self):
        return self.__seed

    def print_instance(self):
        print(self.__seed)
        print(self.__profiling_dir)
        #print(self.__altitude)
        #print(self.__formation)
        #print(self.__ecm)
        #print(self.__incAlt_state)
        #print(self.__decAlt_state)
        #print(self.__incAlt2_state)
        #print(self.__decAlt2_state)
        print(self.__features)
        print(self.__label)

    def get_features(self):
        #print("Original features = ", self.__features)
        features = list()
        #if (self.__compare_past_workload_for_similarity == 1):
        features = self.__features
        #else:
        #    i = 0
            #print("features count = ", len(self.__features))
        #    while (i < 11):
                #print("index = ", i)
        #        features.append(self.__features[i])
        #        i = i + 1

        #    i = len(self.__features) - 6
            
        #    while (i < len(self.__features)):
        #        features.append(self.__features[i])
        #        i = i + 1
            
        #print("features = ", features)
        return features
        
    def get_label(self):
        return self.__label

    def get_actual_label(self):
        return self.__actual_label

class ProblemDatabase:
    __slots__ = ['__features_list', '__labels_list', '__problems']
    
    def __init__(self):
        #print("Inside ProblemDatabase::init")
        self.__features_list = list()
        self.__labels_list = list()
        self.__problems = list()

    def get_features_list(self):
        return self.__features_list

    def get_label_list(self):
        return self.__labels_list

    def get_problems(self):
        return self.__problems

    def get_seed_problems(self, seed):
        seed_problems = list()

        for problem in self.__problems:
            if (problem.get_seed() == seed):
                seed_problems.append(problem)

        return seed_problems

    def add_data(self, problem):
        #features_temp = list()
        #features_temp.append(problem.get_features())

        self.__features_list.append(problem.get_features())
        #print(self.__features_list)
        self.__labels_list.append(problem.get_label())
        self.__problems.append(problem)
    
    def get_db_size(self):
        return len(self.__problems)

if __name__ == '__main__':
    pb_db = ProblemDatabase()


    threat = list()
    target = list()
   
    i = 1 

    while (i < 47):
        target.append(i)
        i = i + 1

    while (i < 93):
        threat.append(i)
        i = i + 1

    print("target = ", target)
    print("threat = ", threat)

    features = ProblemFeatures(1, 2, 3, 4, 5, 6, 7, target, threat)
    features.print_features()
    pb_inst = ProblemInstance(10, "20", features, 1)
    pb_inst.print_instance()
    pb_db.add_data(pb_inst)
    pb_db.add_data(pb_inst)
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    X = pb_db.get_features_list()
    y = pb_db.get_label_list()

    print("X = ", X)
    print("y = ", y)
    neigh.fit(X, y)
    
