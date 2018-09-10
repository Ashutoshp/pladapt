/*******************************************************************************
 * PLA Adaptation Manager
 *
 * Copyright 2017 Carnegie Mellon University. All Rights Reserved.
 * 
 * NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING
 * INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON
 * UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS
 * TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE
 * OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE
 * MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND
 * WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
 *
 * Released under a BSD-style license, please see license.txt or contact
 * permission@sei.cmu.edu for full terms.
 *
 * [DISTRIBUTION STATEMENT A] This material has been approved for public release
 * and unlimited distribution. Please see Copyright notice for non-US Government
 * use and distribution.
 ******************************************************************************/
#include <pladapt/Utils.h>
#include <math.h>
#include <sys/stat.h>
#include <string.h>
#include <stdexcept>
#include <iostream>


DumpPlanningProblems* DumpPlanningProblems::mDumpPlanningProblems = NULL;

// Generate fast plan all the time
// Generate features

int createDirectory(const char* path) {
    const int dirErr = mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    if (dirErr == -1) {
        throw runtime_error("error createDirectory");
    }

    return dirErr;
}

string createTempDirectory(const char* tempDirTemp) {
    char tempDirTemplate[2048];
    strcpy(tempDirTemplate, tempDirTemp);

    // create temp directory
    char* tempDir = mkdtemp(tempDirTemplate);

    if (!tempDir) {
        // TODO improve error handling
        throw runtime_error("error create_temp_directory mkdtemp");
    }

    return tempDir;
}

void copyFile(string fileNameFrom, string fileNameTo) {
    //cout << "copyFile = " << fileNameFrom << "  " << fileNameTo << endl;
    std::ifstream in (fileNameFrom.c_str());
    std::ofstream out (fileNameTo.c_str());
    out << in.rdbuf();
    out.close();
    in.close();
}


//void copy_files_from_directory(const string& source, const string& destination) {

//}
DumpPlanningProblems::DumpPlanningProblems(const string& location, const int seed) : mLocation(location),
                                               mSeed(seed),
                                               mModelTemplate("modelXXXXXX"),
                                               mSpecFileName("ttimemodel.prism"),
                                               mStatesFileName("result.sta"),
                                               mLabelsFileName("result.lab"),
                                               mAdversaryFileName("result.adv"),
                                               mFastDirName("fast"),
                                               mSlowDirName("slow"),
                                               mTempDirTemplate(mLocation +  "/" + mModelTemplate) {

    mFeaturesFileName = to_string(seed) + ".csv";
}

DumpPlanningProblems::~DumpPlanningProblems() {


}

void DumpPlanningProblems::writeHeader(ofstream& fout) {
    fout << "Seed" << "," << "altitude"
            << "," << "formation"
            << "," << "ecm"
            << "," << "ecmOn_used"
            << "," << "ecmOn_go"
            << "," << "ecmOff_used"
            << "," << "ecmOff_go"
            << "," << "goTight_used"
            << "," << "goTight_go"
            << "," << "goLoose_used"
            << "," << "goLoose_go"
            << "," << "incAlt_state"
            << "," << "incAlt_go"
            << "," << "decAlt_state"
            << "," << "decAlt_go"
            << "," << "incAlt2_state"
            << "," << "incAlt2_go"
            << "," << "decAlt2_state"
            << "," << "decAlt2_go"
            << "," << "satisfied"
            << "," << "targetDetected"
			<< "," << "envState"
            /*<< "," << "RA 9"
            << "," << "RA 10"
            << "," << "RA 11"
            << "," << "RA 12"
            << "," << "RA 13"
            << "," << "RA 14"
            << "," << "RA 15"
            << "," << "RA 16"
            << "," << "RA 17"
            << "," << "RA 18"
            << "," << "RA 19"
            << "," << "RA 20"
            << "," << "Predicted"*/
            << "," << "Use Reactive"
            << "\n";
}

void DumpPlanningProblems::copyFileFromFast(const string& source, const string& destination) {
    //cout << "DumpPlanningProblems::copyFileFromFast" << endl;
    string srcModelPath = source + "/" + mSpecFileName;
    string srcAdversaryPath = source + "/" + mAdversaryFileName;
    string srcStatesPath = source + "/" + mStatesFileName;
    string srcLabelsPath = source + "/" + mLabelsFileName;

    string destModelPath = destination + "/" + mSpecFileName;
    string destAdversaryPath = destination + "/" + mAdversaryFileName;
    string destStatesPath = destination + "/" + mStatesFileName;
    string destLabelsPath = destination + "/" + mLabelsFileName;

    // specification file .... though not required
    copyFile(srcModelPath, destModelPath);

    // label file
    copyFile(srcAdversaryPath, destAdversaryPath);

    // adversary file
    copyFile(srcStatesPath, destStatesPath);

    // states file
    copyFile(srcLabelsPath, destLabelsPath);
}

void DumpPlanningProblems::copyFileFromSlow(const string& source, const string& destination) {
    //cout << "DumpPlanningProblems::copyFileFromSlow" << endl;
    // spec file
    string srcModelPath = source + "/" + mSpecFileName;
    string destModelPath = destination + "/" + mSpecFileName;

    copyFile(srcModelPath, destModelPath);
}


void DumpPlanningProblems::writeInitialStateVariables(ofstream& fout, const State& currentState) {
    //cout << "DumpPlanningProblems::writeInitialStateVariables" << endl;
	fout << "," << currentState.altitude
			<< "," << currentState.formation
			<< "," << currentState.ecm
			<< "," << currentState.ecmOn_used
			<< "," << currentState.ecmOn_go
			<< "," << currentState.ecmOff_used
			<< "," << currentState.ecmOff_go
			<< "," << currentState.goTight_used
			<< "," << currentState.goTight_go
			<< "," << currentState.goLoose_used
			<< "," << currentState.goLoose_used
			<< "," << currentState.incAlt_state
			<< "," << currentState.incAlt_go
			<< "," << currentState.decAlt_state
			<< "," << currentState.decAlt_go
			<< "," << currentState.incAlt2_state
			<< "," << currentState.incAlt2_go
			<< "," << currentState.decAlt2_state
			<< "," << currentState.decAlt2_go
			<< "," << currentState.satisfied
			<< "," << currentState.targetDetected
			<< "," << currentState.env_state;
}

void DumpPlanningProblems::writeData(const string& destinationDir,
        const string& reactivePlanDir,
        const string& deliberativePlanDir,
        const State& currentState,
        const string& envModel,
        double classifierLabel) {

    static bool headerWritten = false;
    //cout << "DumpPlanningProblems::writeData" << endl;

    // Open file
    string filePath = mLocation + "/" + mFeaturesFileName;

    // Append to the file
    ofstream fout(filePath.c_str(), std::ofstream::out | std::ofstream::app);

    if (!headerWritten) {
        writeHeader(fout);
        headerWritten = true;
    }

    fout << mSeed << "," << destinationDir;
            //<< "," << reactivePlanDir
            //<< "," << deliberativePlanDir;

    // Add initial state variables
    writeInitialStateVariables(fout, currentState);

    // Now add arrival rates
    /*std::vector<double>::const_iterator itr = envModel.begin();

    while (itr != envModel.end()) {
        fout << "," << *itr;
        ++itr;
    }*/

    //fout << "," << classifierLabel;
    fout << endl;

    // Close file
    fout.close();
}

void DumpPlanningProblems::copySampleProblems(
        const string& reactivePlanDir,
        const string& deliberativePlanDir,
        const State& currentState,
        const string& envModel,
        double classifierLabel) {
    // Create parent directory
    //char tempDirTemplate[] = "modelXXXXXX";
    //string tempDirTemplate = location +  "/" + mModelTemplate;
    //cout << "DumpPlanningProblems::copySampleProblems reactivePlanDir = " << reactivePlanDir << endl;
    string path = createTempDirectory(mTempDirTemplate.c_str());

    string fastPath = path +  "/" + mFastDirName;
    string slowPath = path + "/" + mSlowDirName;
    //string features_path = path + "/features";

    // Create fast directory
    createDirectory(fastPath.c_str());

    // Create slow directory
    createDirectory(slowPath.c_str());

    // Copy fast
    copyFileFromFast(reactivePlanDir, fastPath);

    // Copy slow
    copyFileFromSlow(deliberativePlanDir, slowPath);

    // Write features
    writeData(path, reactivePlanDir, deliberativePlanDir, currentState, envModel, classifierLabel);
}

namespace pladapt {

unsigned tacticLatencyToPeriods(double tacticLatency, double evaluationPeriod) {
    return ceil(tacticLatency / evaluationPeriod);
}


} /* namespace pladapt */


