/*
 * DebugFileInfo.h
 *
 *  Created on: Jul 18, 2018
 *      Author: ashutosp
 */

#ifndef UTIL_DEBUGFILEINFO_H_
#define UTIL_DEBUGFILEINFO_H_

using namespace std;

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

class DebugFileInfo {
private:
    static DebugFileInfo* mDebugFileInfo;
    ofstream* mFoutP;
    const int mSeed;
    const string mParentDirectory;
    string mMode;
    const string mDetailedDebugFile;

    DebugFileInfo(const int seed, const string& parentDirectory, const string& mode);
    void CreateLogsDir() const;
    string GetLogsDirPath() const;
    string GetSeedDirPath() const;

public:
    static DebugFileInfo* getInstance(const int seed = -1,
            const char* parentDir = "", const char* mode = "Fast");

    string GetDebugFilePath() const;

    void write(const string& output) const;
    void write(const int& output) const;
    void closeWriter();
    void cleanFiles();

    virtual ~DebugFileInfo();
};

#endif /* UTIL_DEBUGFILEINFO_H_ */
