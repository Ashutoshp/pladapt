/*
 * ClassifyProblemInstance.h
 *
 *  Created on: Mar 13, 2018
 *      Author: ashutosp
 */

#ifndef MACHINELEARNING_CLASSIFYPROBLEMINSTANCE_H_
#define MACHINELEARNING_CLASSIFYPROBLEMINSTANCE_H_

#include <stdio.h>
#include <iostream>
#include <ctime>
#include "/usr/include/python3.5m/Python.h"
#include <string>
#include <vector>
#include "../examples/dart/dartam/include/dartam/DartConfiguration.h"

using namespace std;
using namespace dart;
using namespace am2;


class ClassifyProblemInstance {
private:
    static ClassifyProblemInstance* mClassifyProblemInstance;

    ClassifyProblemInstance(const string& problem_db_file, string source_file);

    void create_classifier(string problem_db_file, string source_file);
    //unsigned getEstimatorCount(string source_file) const;

    PyObject *mClassifierObject;

public:
    typedef vector<double> TimeSeries;

    static ClassifyProblemInstance* getInstance(string problem_db_file = "", string source_file = "") {
        if (mClassifyProblemInstance == NULL) {
            mClassifyProblemInstance = new ClassifyProblemInstance(problem_db_file, source_file);
        }

        return mClassifyProblemInstance;
    }

    static void Clean();

    double useReactive(const DartConfiguration* config,
        const std::vector<double>& targetPredictions,
        const std::vector<double>& threatPredictions) const;
    virtual ~ClassifyProblemInstance();
};


#endif /* MACHINELEARNING_CLASSIFYPROBLEMINSTANCE_H_ */
