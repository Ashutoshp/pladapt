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
#ifndef UTILS_H_
#define UTILS_H_

#include <string>
#include <vector>
#include <fstream>
#include <pladapt/State.h>

namespace pladapt {
/**
 * Converts tactic latency to periods.
 *
 * This is to make the conversion uniform and easy to change globally,
 * for example switching from ceil to round.
 */
unsigned tacticLatencyToPeriods(double tacticLatency, double evaluationPeriod);
} /* namespace pladapt */


using namespace std;
using namespace dart::am2;

int create_directory(const char* path);
string create_temp_directory(const char* tempDirTemp);
//void copy_files_from_directory(const string& source, const string& destination);
void copy_file(const char* fileNameFrom, const char* fileNameTo);
//void copySampleProblems(const string& reactive_plan_dir,
//        const string& deliberative_plan_dir, const string& location, const std::vector<double>& features);
//void copy_file_from_fast(const string& source, const string& destination);
//void copy_file_from_slow(const string& source, const string& destination);
void test_utils(string location);

class DumpPlanningProblems {

private:
    static DumpPlanningProblems* mDumpPlanningProblems;
    const string mLocation;
    const int mSeed;
    const string mModelTemplate;
    const string mSpecFileName;
    const string mStatesFileName;
    const string mLabelsFileName;
    const string mAdversaryFileName;
    const string mFastDirName;
    const string mSlowDirName;
    const string mTempDirTemplate;
    string mFeaturesFileName;

    DumpPlanningProblems(const string& location, const int seed);
    void writeHeader(ofstream& fout);


public:
    static DumpPlanningProblems* get_instance(const string& location, const int seed) {
        if (mDumpPlanningProblems == NULL) {
            mDumpPlanningProblems = new DumpPlanningProblems(location, seed);
        }

        return mDumpPlanningProblems;
    }

    void copySampleProblems(
            const string& reactivePlanDir,
            const string& deliberativePlanDir,
            const State& currentState,
            const string& arrivalRates,
			double classifierLabel);

    void copyFileFromFast(const string& source, const string& destination);
    void copyFileFromSlow(const string& source, const string& destination);
    void writeInitialStateVariables(ofstream& fout, const State& currentState);
    void writeData(const string& destinationDir, const string& reactivePlanDir,
            const string& deliberativePlanDir, const State& currentState,
            const string& arrivalRates, double classifierLabel);

    ~DumpPlanningProblems();
};


#endif /* UTILS_H_ */
