//===============================================================================
// Name        : ClassifyProblemInstance.cpp
// Author      : Ashutosh Pandey
// Version     :
// Copyright   : Your copyright notice
// Description : Bridge between C++ and Python
// ===============================================================================

#include <pladapt/ClassifyProblemInstance.h>
#include <fstream>
#include <pladapt/Utils.h>
#include "../examples/dart/dartam/include/dartam/DebugFileInfo.h"

ClassifyProblemInstance* ClassifyProblemInstance::mClassifyProblemInstance = NULL;

ClassifyProblemInstance::ClassifyProblemInstance(const string& problem_db_file,
        string source_file) : mClassifierObject (NULL) {
    // TODO Auto-generated constructor stub
    create_classifier(problem_db_file, source_file);
}

void ClassifyProblemInstance::Clean() {
    if (mClassifyProblemInstance != NULL) {
        delete mClassifyProblemInstance;
    }

    Py_Finalize();
}

ClassifyProblemInstance::~ClassifyProblemInstance() {
    // TODO Auto-generated destructor stub
    Py_DECREF(mClassifierObject);
    mClassifierObject = NULL;
}

/*unsigned ClassifyProblemInstance::getEstimatorCount(string source_file) const {
    string db_file = omnetpp::getSimulation()->getSystemModule()->par(
                        "pathToTraceEstimatorFile").stringValue();
    //string trace_name = omnetpp::getSimulation()->getSystemModule()->par(
    //        "interArrivalsFile").stringValue();

    EstimatorReader estimator_reader = EstimatorReader(db_file);
    return estimator_reader.GetEstimatorCount(source_file);
}*/

void ClassifyProblemInstance::create_classifier(string problem_db_file, string source_file) {
    PyObject *module, *dict, *python_class;
    const char* classifierScript = "DartDecisionTreeClassifier";

    char* pPath = getenv( "PYTHONPATH" );

    if (pPath!=NULL) {
        printf ("The PYTHONPATH path is: %s\n", pPath);
    } else {
        assert(false);
    }

    Py_Initialize();

    module = PyImport_ImportModule(classifierScript);
    if(module == NULL || PyErr_Occurred() != NULL) {
        PyErr_Print();
        assert(module != NULL);
    }
    //printf("1\n");
    fflush(stdout);

    // Deleting memory for object pName
    //Py_DECREF(module_name);
    //printf("2\n");
    fflush(stdout);

    //printf("3\n");
    fflush(stdout);
    assert(module != NULL);

    if (module != NULL) {
        fflush(stdout);
        //printf("Found\n");
        fflush(stdout);
        //printf("4\n");
        fflush(stdout);

        // dict is a borrowed reference.
        dict = PyModule_GetDict(module);
        //printf("5\n");
        fflush(stdout);

        if (dict == NULL) {
            PyErr_Print();
            std::cerr << "Fails to get the dictionary.\n";
            Py_DECREF(module);
            return;
        }

        //printf("6\n");
        fflush(stdout);
        Py_DECREF(module);

        // Builds the name of a callable class
        python_class = PyDict_GetItemString(dict, "DartDecisionTreeClassifier");
        //printf("7\n");

        if (python_class == NULL) {
            PyErr_Print();
            std::cerr << "Fails to get the Python class.\n";
            Py_DECREF(dict);
            return;
        }

        Py_DECREF(dict);
        //printf("8\n");
        fflush(stdout);

        // Creates an instance of the class
        if (PyCallable_Check(python_class)) {
            //object = PyObject_CallFunction(python_class, "abc", "1");
            PyObject *args;

            //long int estimatorsCount =  85;
            long int depth =  100;
            args = Py_BuildValue("ssi", problem_db_file.c_str(), source_file.c_str(), depth);
            mClassifierObject = PyObject_CallObject(python_class, args);
            //string file_name = DebugFileInfo::getInstance()->GetDebugFilePath();
                 //   "/home/ashutosp/Dropbox/regression/HP_triggers_arrival_rate";

            //std::ofstream myfile;
            //myfile.open(file_name, ios::app);
            //myfile << "estimatorsCount = " << estimatorsCount << endl;
            //myfile.close();

            //string trace_estimator = "/home/ashutosp/Dropbox/regression/trace_estimator.csv";
            //myfile.open(trace_estimator, std::ios::app);
            //myfile << source_file << "," << estimatorsCount << std::endl;
            //myfile.close();

            if(mClassifierObject == NULL || PyErr_Occurred() != NULL) {
                PyErr_Print();
                assert(mClassifierObject != NULL);
            }

            PyObject *success = PyObject_CallMethod(mClassifierObject, "train", NULL);

            if(success == NULL || PyErr_Occurred() != NULL) {
                PyErr_Print();
            }

            Py_DECREF(args);
            Py_DECREF(python_class);
        } else {
            std::cout << "Cannot instantiate the Python class" << std::endl;
            Py_DECREF(python_class);
            return;
        }
    }
}

double ClassifyProblemInstance::useReactive(const DartConfiguration* config,
        const std::vector<double>& targetPredictions,
        const std::vector<double>& threatPredictions) const {
    double useReactive = -1;
    std::cout << "ClassifyProblemInstance::useReactive" << std::endl;
    // TODO pass as integers/floats rather than string
    unsigned altitude = config->getAltitudeLevel();
    unsigned formation = config->getFormation();
    unsigned ecm = config->getEcm();
    unsigned ttcIncAlt = config->getTtcIncAlt();
    unsigned ttcDecAlt = config->getTtcDecAlt();
    unsigned ttcIncAlt2 = config->getTtcIncAlt2();
    unsigned ttcDecAlt2 = config->getTtcDecAlt2();

    PyObject *value = PyObject_CallMethod(mClassifierObject, "classifier_predict",
            "(sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss)", 
            to_string(altitude).c_str(),
            to_string(formation).c_str(), to_string(ecm).c_str(),
            to_string(ttcIncAlt).c_str(), to_string(ttcDecAlt).c_str(),
            to_string(ttcIncAlt2).c_str(), to_string(ttcDecAlt2).c_str(),
            to_string(targetPredictions[0]).c_str(), to_string(threatPredictions[0]).c_str(),
            to_string(targetPredictions[1]).c_str(), to_string(threatPredictions[1]).c_str(),
            to_string(targetPredictions[2]).c_str(), to_string(threatPredictions[2]).c_str(),
            to_string(targetPredictions[3]).c_str(), to_string(threatPredictions[3]).c_str(),
            to_string(targetPredictions[4]).c_str(), to_string(threatPredictions[4]).c_str(),
            to_string(targetPredictions[5]).c_str(), to_string(threatPredictions[5]).c_str(),
            to_string(targetPredictions[6]).c_str(), to_string(threatPredictions[6]).c_str(),
            to_string(targetPredictions[7]).c_str(), to_string(threatPredictions[7]).c_str(),
            to_string(targetPredictions[8]).c_str(), to_string(threatPredictions[8]).c_str(),
            to_string(targetPredictions[9]).c_str(), to_string(threatPredictions[9]).c_str(),
            to_string(targetPredictions[10]).c_str(), to_string(threatPredictions[10]).c_str(),
            to_string(targetPredictions[11]).c_str(), to_string(threatPredictions[11]).c_str(),
            to_string(targetPredictions[12]).c_str(), to_string(threatPredictions[12]).c_str(),
            to_string(targetPredictions[13]).c_str(), to_string(threatPredictions[13]).c_str(),
            to_string(targetPredictions[14]).c_str(), to_string(threatPredictions[14]).c_str(),
            to_string(targetPredictions[15]).c_str(), to_string(threatPredictions[15]).c_str(),
            to_string(targetPredictions[16]).c_str(), to_string(threatPredictions[16]).c_str(),
            to_string(targetPredictions[17]).c_str(), to_string(threatPredictions[17]).c_str(),
            to_string(targetPredictions[18]).c_str(), to_string(threatPredictions[18]).c_str(),
            to_string(targetPredictions[19]).c_str(), to_string(threatPredictions[19]).c_str(),
            to_string(targetPredictions[20]).c_str(), to_string(threatPredictions[20]).c_str(),
            to_string(targetPredictions[21]).c_str(), to_string(threatPredictions[21]).c_str(),
            to_string(targetPredictions[22]).c_str(), to_string(threatPredictions[22]).c_str(),
            to_string(targetPredictions[23]).c_str(), to_string(threatPredictions[23]).c_str(),
            to_string(targetPredictions[24]).c_str(), to_string(threatPredictions[24]).c_str(),
            to_string(targetPredictions[25]).c_str(), to_string(threatPredictions[25]).c_str(),
            to_string(targetPredictions[26]).c_str(), to_string(threatPredictions[26]).c_str(),
            to_string(targetPredictions[27]).c_str(), to_string(threatPredictions[27]).c_str(),
            to_string(targetPredictions[28]).c_str(), to_string(threatPredictions[28]).c_str(),
            to_string(targetPredictions[29]).c_str(), to_string(threatPredictions[29]).c_str(),
            to_string(targetPredictions[30]).c_str(), to_string(threatPredictions[30]).c_str(),
            to_string(targetPredictions[31]).c_str(), to_string(threatPredictions[31]).c_str(),
            to_string(targetPredictions[32]).c_str(), to_string(threatPredictions[32]).c_str(),
            to_string(targetPredictions[33]).c_str(), to_string(threatPredictions[33]).c_str(),
            to_string(targetPredictions[34]).c_str(), to_string(threatPredictions[34]).c_str(),
            to_string(targetPredictions[35]).c_str(), to_string(threatPredictions[35]).c_str(),
            to_string(targetPredictions[36]).c_str(), to_string(threatPredictions[36]).c_str(),
            to_string(targetPredictions[37]).c_str(), to_string(threatPredictions[37]).c_str(),
            to_string(targetPredictions[38]).c_str(), to_string(threatPredictions[38]).c_str(),
            to_string(targetPredictions[39]).c_str(), to_string(threatPredictions[39]).c_str(),
            to_string(targetPredictions[40]).c_str(), to_string(threatPredictions[40]).c_str(),
            to_string(targetPredictions[41]).c_str(), to_string(threatPredictions[41]).c_str(),
            to_string(targetPredictions[42]).c_str(), to_string(threatPredictions[42]).c_str(),
            to_string(targetPredictions[43]).c_str(), to_string(threatPredictions[43]).c_str(),
            to_string(targetPredictions[44]).c_str(), to_string(threatPredictions[44]).c_str(),
            to_string(targetPredictions[45]).c_str(), to_string(threatPredictions[45]).c_str());


    if (value == NULL) {
        PyErr_Print();
        assert(false);
    } else {
        printf("Result of call: %f\n", PyFloat_AsDouble(value));
        //useReactive = (PyFloat_AsDouble(value) == 1.0) ? true : false;
        useReactive = PyFloat_AsDouble(value);
        Py_DECREF(value);
    }

    //string file_name = DebugFileInfo::getInstance()->GetDebugFilePath();
    //std::ofstream myfile;

    //myfile.open(file_name, ios::app);
    string str = "Features = [" + to_string(altitude) + ", " + to_string(formation) + ", "
            + to_string(ecm) + ", " + to_string(ttcIncAlt) + ", " + to_string(ttcDecAlt) + ", "
            + to_string(ttcIncAlt2) + ", " + to_string(ttcDecAlt2) + ", "
            + to_string(targetPredictions[0]) + ", " + to_string(threatPredictions[0]) + ", " 
            + to_string(targetPredictions[1]) + ", " + to_string(threatPredictions[1]) + ", "
            + to_string(targetPredictions[2]) + ", " + to_string(threatPredictions[2]) + ", "
            + to_string(targetPredictions[3]) + ", " + to_string(threatPredictions[3]) + ", "
            + to_string(targetPredictions[4]) + ", " + to_string(threatPredictions[4]) + ", "
            + to_string(targetPredictions[5]) + ", " + to_string(threatPredictions[5]) + ", "
            + to_string(targetPredictions[6]) + ", " + to_string(threatPredictions[6]) + ", "
            + to_string(targetPredictions[7]) + ", " + to_string(threatPredictions[7]) + ", "
            + to_string(targetPredictions[8]) + ", " + to_string(threatPredictions[8]) + ", "
            + to_string(targetPredictions[9]) + ", " + to_string(threatPredictions[9]) + ", "
            + to_string(targetPredictions[10]) + ", " + to_string(threatPredictions[10]) + ", "
            + to_string(targetPredictions[11]) + ", " + to_string(threatPredictions[11]) + ", "
            + to_string(targetPredictions[12]) + ", " + to_string(threatPredictions[12]) + ", "
            + to_string(targetPredictions[13]) + ", " + to_string(threatPredictions[13]) + ", "
            + to_string(targetPredictions[14]) + ", " + to_string(threatPredictions[14]) + ", "
            + to_string(targetPredictions[15]) + ", " + to_string(threatPredictions[15]) + ", "
            + to_string(targetPredictions[16]) + ", " + to_string(threatPredictions[16]) + ", "
            + to_string(targetPredictions[17]) + ", " + to_string(threatPredictions[17]) + ", "
            + to_string(targetPredictions[18]) + ", " + to_string(threatPredictions[18]) + ", "
            + to_string(targetPredictions[19]) + ", " + to_string(threatPredictions[19]) + ", "
            + to_string(targetPredictions[20]) + ", " + to_string(threatPredictions[20]) + ", "
            + to_string(targetPredictions[21]) + ", " + to_string(threatPredictions[21]) + ", "
            + to_string(targetPredictions[22]) + ", " + to_string(threatPredictions[22]) + ", "
            + to_string(targetPredictions[23]) + ", " + to_string(threatPredictions[23]) + ", "
            + to_string(targetPredictions[24]) + ", " + to_string(threatPredictions[24] )+ ", "
            + to_string(targetPredictions[25]) + ", " + to_string(threatPredictions[25]) + ", "
            + to_string(targetPredictions[26]) + ", " + to_string(threatPredictions[26]) + ", "
            + to_string(targetPredictions[27]) + ", " + to_string(threatPredictions[27]) + ", "
            + to_string(targetPredictions[28]) + ", " + to_string(threatPredictions[28]) + ", "
            + to_string(targetPredictions[29]) + ", " + to_string(threatPredictions[29]) + ", "
            + to_string(targetPredictions[30]) + ", " + to_string(threatPredictions[30]) + ", "
            + to_string(targetPredictions[31]) + ", " + to_string(threatPredictions[31]) + ", "
            + to_string(targetPredictions[32]) + ", " + to_string(threatPredictions[32]) + ", "
            + to_string(targetPredictions[33]) + ", " + to_string(threatPredictions[33]) + ", "
            + to_string(targetPredictions[34]) + ", " + to_string(threatPredictions[34]) + ", "
            + to_string(targetPredictions[35]) + ", " + to_string(threatPredictions[35]) + ", "
            + to_string(targetPredictions[36]) + ", " + to_string(threatPredictions[36]) + ", "
            + to_string(targetPredictions[37]) + ", " + to_string(threatPredictions[37]) + ", "
            + to_string(targetPredictions[38]) + ", " + to_string(threatPredictions[38]) + ", "
            + to_string(targetPredictions[39]) + ", " + to_string(threatPredictions[39]) + ", "
            + to_string(targetPredictions[40]) + ", " + to_string(threatPredictions[40]) + ", "
            + to_string(targetPredictions[41]) + ", " + to_string(threatPredictions[41]) + ", "
            + to_string(targetPredictions[42]) + ", " + to_string(threatPredictions[42]) + ", "
            + to_string(targetPredictions[43]) + ", " + to_string(threatPredictions[43]) + ", "
            + to_string(targetPredictions[44]) + ", " + to_string(threatPredictions[44]) + ", "
            + to_string(targetPredictions[45]) + ", " + to_string(threatPredictions[45]) + "]\n";

    DebugFileInfo::getInstance()->write(str);
    DebugFileInfo::getInstance()->write("Prediction = ", false);
    DebugFileInfo::getInstance()->write(unsigned(PyFloat_AsDouble(value)));

    return useReactive;
}
