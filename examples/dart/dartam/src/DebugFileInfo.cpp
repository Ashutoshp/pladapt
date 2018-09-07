/*
 * DebugFileInfo.cc
 *
 *  Created on: Jul 18, 2018
 *      Author: ashutosp
 */

#include <assert.h>
#include <boost/filesystem.hpp>
#include <dartam/DebugFileInfo.h>

DebugFileInfo* DebugFileInfo::mDebugFileInfo = NULL;

DebugFileInfo::DebugFileInfo(const int seed, const string& parentDirectory, const string& mode) :
          mSeed(seed),
          mParentDirectory(parentDirectory),
          mMode(mode),
          mDetailedDebugFile("debug") {
    // TODO Auto-generated constructor stub

    if (mode == "pg") mMode = "ProbGeneration";
    else if (mode == "cb") mMode = "Hybrid";
    else if (mode == "ml0") mMode = "Hybrid_IBL0";
    else if (mode == "ml1") mMode = "Hybrid_IBL1";
    else if (mode == "slow") mMode = "Slow";
    else if (mode == "fast") mMode = "Fast";
    else if (mode == "si") mMode = "SlowInstant";
    else assert(false);

    CreateLogsDir();
    mFoutP = new ofstream(GetDebugFilePath().c_str());

	if (mFoutP == NULL) {
		cout << "Could not open log file  "
				<< GetDebugFilePath() << endl;
	}
}

DebugFileInfo::~DebugFileInfo() {
    // TODO Auto-generated destructor stub
	mFoutP->close();

	if (mFoutP != NULL) delete mFoutP;
	mDebugFileInfo = NULL;
}

DebugFileInfo* DebugFileInfo::getInstance(const int seed, const char* parentDir, const char* mode) {
    if (mDebugFileInfo == NULL) {
        mDebugFileInfo = new DebugFileInfo(seed, parentDir, mode);
    }

    return mDebugFileInfo;
}

void DebugFileInfo::write(const string& output) const {
	*mFoutP << output << endl;
}

void DebugFileInfo::write(const int& output) const {
	*mFoutP << output << endl;
}

void DebugFileInfo::closeWriter() {
	delete mDebugFileInfo;
}

void DebugFileInfo::CreateLogsDir() const {
    string dir = GetLogsDirPath();
    boost::filesystem::path p(dir);
    boost::system::error_code errCode;
    
    if (!(boost::filesystem::exists(p)
            && boost::filesystem::is_directory(p))) {
        bool result = boost::filesystem::create_directories(
					dir, errCode);

        assert(result);
    }
}

string DebugFileInfo::GetLogsDirPath() const {
    return GetSeedDirPath() + "/" + mMode;
}

string DebugFileInfo::GetSeedDirPath() const {
    return mParentDirectory + "/" + std::to_string(mSeed);
}

string DebugFileInfo::GetDebugFilePath() const {
    string path = "";
    path = GetLogsDirPath() + "/" + mDetailedDebugFile;

    return path;
}
