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
#include <boost/filesystem.hpp>

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
    boost::filesystem::remove(mLocation +  "/" + mFeaturesFileName);    
}

DumpPlanningProblems::~DumpPlanningProblems() {


}

void DumpPlanningProblems::writeHeader(ofstream& fout) {
    fout << "Seed"
            << ", " << "Directory" 
            << ", " << "altitude"
            << ", " << "formation"
            << ", " << "ecm"
            //<< "," << "ecmOn_used"
            //<< "," << "ecmOn_go"
            //<< "," << "ecmOff_used"
            //<< "," << "ecmOff_go"
            //<< "," << "goTight_used"
            //<< "," << "goTight_go"
            //<< "," << "goLoose_used"
            //<< "," << "goLoose_go"
            << ", " << "incAlt_state"
            //<< "," << "incAlt_go"
            << ", " << "decAlt_state"
            //<< "," << "decAlt_go"
            << ", " << "incAlt2_state"
            //<< "," << "incAlt2_go"
            << ", " << "decAlt2_state"
            //<< "," << "decAlt2_go"
            //<< "," << "satisfied"
            //<< "," << "targetDetected"
			//<< "," << "envState"
            << ", " << "TR1"
            << ", " << "TH1"
            << ", " << "TR2"
            << ", " << "TH2"
            << ", " << "TR3"
            << ", " << "TH3"
            << ", " << "TR4"
            << ", " << "TH4"
            << ", " << "TR5"
            << ", " << "TH5"
            << ", " << "TR6"
            << ", " << "TH6"
            << ", " << "TR7"
            << ", " << "TH7"
            << ", " << "TR8"
            << ", " << "TH8"
            << ", " << "TR9"
            << ", " << "TH9"
            << ", " << "TR10"
            << ", " << "TH10"
            << ", " << "TR11"
            << ", " << "TH11"
            << ", " << "TR12"
            << ", " << "TH12"
            << ", " << "TR13"
            << ", " << "TH13"
            << ", " << "TR14"
            << ", " << "TH14"
            << ", " << "TR15"
            << ", " << "TH15"
            << ", " << "TR16"
            << ", " << "TH16"
            << ", " << "TR17"
            << ", " << "TH17"
            << ", " << "TR18"
            << ", " << "TH18"
            << ", " << "TR19"
            << ", " << "TH19"
            << ", " << "TR20"
            << ", " << "TH20"
            << ", " << "TR21"
            << ", " << "TH21"
            << ", " << "TR22"
            << ", " << "TH22"
            << ", " << "TR23"
            << ", " << "TH23"
            << ", " << "TR24"
            << ", " << "TH24"
            << ", " << "TR25"
            << ", " << "TH25"
            << ", " << "TR26"
            << ", " << "TH26"
            << ", " << "TR27"
            << ", " << "TH27"
            << ", " << "TR28"
            << ", " << "TH28"
            << ", " << "TR29"
            << ", " << "TH29"
            << ", " << "TR30"
            << ", " << "TH30"
            << ", " << "TR31"
            << ", " << "TH31"
            << ", " << "TR32"
            << ", " << "TH32"
            << ", " << "TR33"
            << ", " << "TH33"
            << ", " << "TR34"
            << ", " << "TH34"
            << ", " << "TR35"
            << ", " << "TH35"
            << ", " << "TR36"
            << ", " << "TH36"
            << ", " << "TR37"
            << ", " << "TH37"
            << ", " << "TR38"
            << ", " << "TH38"
            << ", " << "TR39"
            << ", " << "TH39"
            << ", " << "TR40"
            << ", " << "TH40"
            << ", " << "TR41"
            << ", " << "TH41"
            << ", " << "TR42"
            << ", " << "TH42"
            << ", " << "TR43"
            << ", " << "TH43"
            << ", " << "TR44"
            << ", " << "TH44"
            << ", " << "TR45"
            << ", " << "TH45"
            << ", " << "TR46"
            << ", " << "TH46"
            //<< "," << "Predicted"
            << ", " << "Use Reactive"
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


void DumpPlanningProblems::writeInitialStateVariables(ofstream& fout, const DartConfiguration* config) {
    //cout << "DumpPlanningProblems::writeInitialStateVariables" << endl;
    fout << ", " << config->getAltitudeLevel()
         << ", " << config->getFormation()
         << ", " << config->getEcm()
         << ", " << config->getTtcIncAlt()
         << ", " << config->getTtcDecAlt()
         << ", " << config->getTtcIncAlt2()
         << ", " << config->getTtcDecAlt2();
}

void DumpPlanningProblems::writeData(const string& destinationDir,
        const string& reactivePlanDir,
        const string& deliberativePlanDir,
        const DartConfiguration* config,
        const pladapt::EnvironmentDTMCPartitioned* envModel,
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
    writeInitialStateVariables(fout, config);

    // Add target and threat probabilities.
    for (unsigned s = 0; s < envModel->getNumberOfStates(); ++s) {
		const auto& envValue = envModel->getStateValue(s);
		unsigned targetProb = envValue.getComponent(0).asDouble();
		unsigned threatProb = envValue.getComponent(1).asDouble();

        fout << ", " << targetProb << ", " << threatProb;
    }

    //fout << "," << classifierLabel;
    fout << endl;

    // Close file
    fout.close();
}

void DumpPlanningProblems::copySampleProblems(
        const string& reactivePlanDir,
        const string& deliberativePlanDir,
        const DartConfiguration* config,
        const pladapt::EnvironmentDTMCPartitioned* envModel,
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
    writeData(path, reactivePlanDir, deliberativePlanDir, config, envModel, classifierLabel);
}

namespace pladapt {

unsigned tacticLatencyToPeriods(double tacticLatency, double evaluationPeriod) {
    return ceil(tacticLatency / evaluationPeriod);
}


} /* namespace pladapt */


