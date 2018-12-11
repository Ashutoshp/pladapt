import sys
from collections import defaultdict
from ProblemDB import *

class DBWrapper:
    def __init__(self, data_file):
        self.__db_file = data_file
        self.__all_seeds = set()
        self.__use_reactive_problems = ProblemDatabase()
        self.__not_use_reactive_problems = ProblemDatabase()
        self.__use_either_problems = ProblemDatabase()
        self.__all_training_problems = ProblemDatabase()

    def get_all_seeds(self):
        return self.__all_seeds

    def get_all_training_problems(self):
        return self.__all_training_problems

    def get_use_reactive_problems(self):
        return self.__use_reactive_problems

    def get_not_reactive_problems(self):
        return self.__not_use_reactive_problems

    def get_use_either_problems(self):
        return self.__use_either_problems

    def read_db(self):
        print ("Reading problem database file from", self.__db_file)
 
        # Open the same file to compare with itself
        db_csvfile = open(self.__db_file, 'r')

        #print (db_csvfile)

        rows = 0
        count = 0
        ignored_trace_problems = 0
        line_no = 1
        #print("xyz")

        for line in db_csvfile:
            rows = rows + 1
            #print("rows = ", rows)

            if (rows == 1):
                # TODO Make sure the first line is a header
                continue

            # Remove newline at the end
            line = line.strip('\n')
            #print(line)

            tokens = line.split(',')
            #print(tokens)

            #tokens = re.split(',', line)
            seed = tokens[0]
            #print("trace = ", trace)
            #if (trace == self.__skip_trace):
            #    continue

            #self.__traces.add(trace)

            profiling_dir = tokens[1]
            #print("profiling_dir = ", profiling_dir)
            #altitude = tokens[2]
            #print("fast_dir = ", fast_dir)
            #formation = tokens[3]
            #print("slow_dir = ", slow_dir)
            features = tokens[2:len(tokens) - 1]
            #print("tokens = ", tokens)
            #print("features = ", features)
            #print("workload = ", features[11:len(features)])
            #workload = features[11:len(features)]
            #workload = self.fix_infinity_values(workload)
            #print("Fixed workload = ", workload)
            i = 0
            j = 9
            targets = list()
            threats = list()

            while(i < 46):
                targets.append(tokens[j])
                j = j + 1
                threats.append(tokens[j])
                j = j + 1
                i = i + 1

            prob_features = ProblemFeatures(features[0], features[1], features[2], \
                    features[3], features[4], features[5], features[6], \
                    targets, threats)
            #prob_features.print_features()
            label = int(tokens[len(tokens) - 1])
            #label.strip()
            #print("label = ", label)

            problemInstance = ProblemInstance(seed, profiling_dir, \
                    prob_features, label)
            #problemInstance.print_instance()

            #if prob_features.get_r20() == float('inf'):
            #    self.add_inf_problem(problemInstance)
            #elif trace in self.__ignored_traces:
                #print("Ignoring trace = ", trace)
            #    self.add_ignored_trace_problem(trace, problemInstance)
            #    self.__ignored_traces_problems.add_data(problemInstance)
            #    ignored_trace_problems = ignored_trace_problems + 1
            #else:
            self.__all_training_problems.add_data(problemInstance)
            self.__all_seeds.add(seed)

            if label == 0:
                self.__not_use_reactive_problems.add_data(problemInstance)
            elif label == 1:
                self.__use_reactive_problems.add_data(problemInstance)
            elif label == 2:
                self.__use_either_problems.add_data(problemInstance)
            else:
                print("Error: label = ", label)
                assert False
            #self.__trace_problem[trace].append(problemInstance)
            #break

        db_csvfile.close()

        #print("Number of inf problems = ", len(self.__inf_problems))
        #print("Number of ignored traces = ", len(self.__ignored_trace_problems))
        #print("Number of ignored traces Problems = ", ignored_trace_problems)
        print("Number of problems with not use_reactive = ", self.__not_use_reactive_problems.get_db_size())
        print("Number of problems with use_reactive = ", self.__use_reactive_problems.get_db_size())
        print("Number of problems with use_either = ", self.__use_either_problems.get_db_size())

        #assert(rows == len(self.__use_reactive_problems.get_db_size() + \
        #        self.__not_use_reactive_problems.get_db_size() + \
        #        self.__use_either_problems.get_db_size() + 1)
        #print(self.__pb_db.get_features_list())
        #print(self.__pb_db.get_label_list())

        #self.fit(self.__pb_db.get_features_list(), self.__pb_db.get_label_list())

        return 1

def main():
    #print (sys.argv[0])
    #print (sys.argv[1])

    db_wrapper = DBWrapper(sys.argv[1])
    db_wrapper.read_db()

if __name__ == "__main__":
    sys.exit(main())    
