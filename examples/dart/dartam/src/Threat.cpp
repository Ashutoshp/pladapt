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

#include <dartam/Threat.h>
#include <algorithm>
#include <dartam/RandomSeed.h>

using namespace std;

namespace dart {
namespace am2 {

Threat::Threat(double range, double destructionFormationFactor, double ecmProbability)
	: range(range),
	  destructionFormationFactor(destructionFormationFactor),
      ecmProbability(ecmProbability),
      randomGenerator(RandomSeed::getNextSeed())
{
   //cout << "Threat::Threat destructionFormationFactor = " << destructionFormationFactor << endl;
}

Threat::~Threat() {
}

double Threat::getProbabilityOfDestruction(const DartConfiguration& config) {
	double probOfDestruction =
			((config.getFormation() == DartConfiguration::Formation::LOOSE) ? 1.0 : (1.0 / destructionFormationFactor))
			* max(0.0, range - (config.getAltitudeLevel() + 1)) / range; // +1 because level 0 is one level above ground

    //std::cout << "config.getFormation() = " << config.getFormation() << std::endl;
    /*std::cout << "destructionFormationFactor = " << destructionFormationFactor << std::endl;
    std::cout << "range = " << range << endl;
    std::cout << "config.getAltitudeLevel() = " << config.getAltitudeLevel() << std::endl;
    std::cout << "config.getFormation() == DartConfiguration::Formation::LOOSE = " 
            << (config.getFormation() == DartConfiguration::Formation::LOOSE) << std::endl;
    std::cout << "max(0.0, range - (config.getAltitudeLevel() + 1)) = " 
            << max(0.0, range - (config.getAltitudeLevel() + 1)) << std::endl;

    std::cout << "config.getEcm() = " << config.getEcm() << std::endl;*/

    //cout << "### Threat::getProbabilityOfDestruction probOfDestruction = " << probOfDestruction << endl;
	// ECM reduces the prob of destruction
	if (config.getEcm()) {
		probOfDestruction *= ecmProbability;
	}

    //cout << "Threat::getProbabilityOfDestruction probOfDestruction = " << probOfDestruction << endl;
	return probOfDestruction;
//	return (config.getAltitudeLevel() + 1 <= threatRange) ? 1.0 : 0.0;
}

bool Threat::isDestroyed(const RealEnvironment& threatEnv,
		const DartConfiguration& config, const Coordinate& location) {
	bool destroyed = false;
	bool threat = threatEnv.isObjectAt(location);
	if (threat) {
		double probOfDestruction = getProbabilityOfDestruction(config);
       //cout << "Threat::isDestroyed = probOfDestruction = " << probOfDestruction << endl;
		double random = uniform(randomGenerator);
       //cout << "Threat::isDestroyed = random = " << random << endl;
		destroyed = (random <= probOfDestruction);
	}
	return destroyed;
}

} /* namespace am2 */
} /* namespace dart */
