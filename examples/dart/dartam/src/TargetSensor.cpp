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

#include <dartam/TargetSensor.h>
#include <algorithm>
#include <dartam/RandomSeed.h>
#include <iostream>

using namespace std;

namespace dart {
namespace am2 {

TargetSensor::TargetSensor(double range, double detectionFormationFactor, double ecmProb)
	: range(range), 
	  detectionFormationFactor(detectionFormationFactor), ecmProbability(ecmProb),
	  randomGenerator(RandomSeed::getNextSeed())
{
}

TargetSensor::~TargetSensor() {
}

double TargetSensor::getProbabilityOfDetection(const DartConfiguration& config) {
	double probOfDetection =
			((config.getFormation() == DartConfiguration::Formation::LOOSE) ? 1.0 : 1 / detectionFormationFactor)
			* max(0.0, range - (config.getAltitudeLevel() + 1)) / range; // +1 because level 0 is one level above ground

    //std::cout << "config.getFormation() = " << config.getFormation() << std::endl;
    //std::cout << "detectionFormationFactor = " << detectionFormationFactor << std::endl;
    //std::cout << "range = " << range << endl;
    //std::cout << "config.getAltitudeLevel() = " << config.getAltitudeLevel() << std::endl;
    //std::cout << "config.getFormation() == DartConfiguration::Formation::LOOSE = " 
    //        << (config.getFormation() == DartConfiguration::Formation::LOOSE) << std::endl;
    //std::cout << "max(0.0, range - (config.getAltitudeLevel() + 1)) = " 
    //        << max(0.0, range - (config.getAltitudeLevel() + 1)) << std::endl;

	// ECM reduces the prob of detection
	if (config.getEcm()) {
		probOfDetection *= ecmProbability;
	}

    //cout << "Threat::getProbabilityOfDestruction probOfDetection = " << probOfDetection << endl;

	return probOfDetection;
//	return (config.getAltitudeLevel() + 1 <= sensorRange) ? 1.0 : 0.0;
}

bool TargetSensor::sense(const DartConfiguration& config, bool targetPresent) {
	bool detected = false;
	if (targetPresent) {
		double probOfDetection = getProbabilityOfDetection(config);

		double random = uniform(randomGenerator);
		detected = (random <= probOfDetection);
	}
	return detected;
}

} /* namespace am2 */
} /* namespace dart */
