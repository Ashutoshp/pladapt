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
#include "Simulation.h"
#include <iostream>
#include <getopt.h>
#include <cstdlib>
#include <dartam/RandomSeed.h>
#include <dartam/DartUtilityFunction.h>
#include <dartam/DebugFileInfo.h>
#include <fstream>

// set this to 1 for testing
#define FIXED2DSPACE 0

using namespace dart::am2;
using namespace std;
//DebugFileInfo* DebugFileInfo::mDebugFileInfo = NULL;

enum ARGS {
	MAP_SIZE,
	SQUARE_MAP,
	NUM_TARGETS,
	NUM_THREATS,
	ALTITUDE_LEVELS,
	THREAT_RANGE,
	THREAT_SENSOR_FPR,
	THREAT_SENSOR_FNR,
	TARGET_SENSOR_FPR,
	TARGET_SENSOR_FNR,
	DL_TARGET_SENSOR_RANGE,
	AUTO_RANGE,
	DECISION_HORIZON,
	OBSERVATION_HORIZON,
	ACCUMULATE_OBSERVATIONS,
	REACH_PATH,
	REACH_MODEL,
	DISTRIB_APPROX,
	CHANGE_ALT_LATENCY,
	NON_LATENCY_AWARE,
	SEED,
	PROBABILITY_BOUND,
	STAY_ALIVE_REWARD,
	NO_FORMATION,
	ECM,
	TWO_LEVEL_TACTICS,
	ADAPT_MGR,
	PRISM_TEMPLATE,
	OPT_TEST,
	HP_MODE,
    ECM_THREAT_PROB,
    ECM_TARGET_PROB,
    ECM_DETECT_PROB,
    ERROR_TOLERANCE,
    DETECTION_FORMATION_FACTOR,
    DESTRUCTION_FORMATION_FACTOR,
#if DART_USE_CE
	,
	CE_NONINCREMENTAL,
	CE_HINT_WEIGHT,
	CE_SAMPLES,
	CE_ALPHA,
	CE_PRECISION,
	CE_MAX_ITERATIONS
#endif
};

static struct option long_options[] = {
	{"map-size", required_argument, 0,  MAP_SIZE },
	{"square-map", no_argument, 0,  SQUARE_MAP },
	{"num-targets", required_argument, 0,  NUM_TARGETS },
	{"num-threats", required_argument, 0,  NUM_THREATS },
	{"altitude-levels", required_argument, 0,  ALTITUDE_LEVELS },
	{"threat-range", required_argument, 0,  THREAT_RANGE },
    {"threat-sensor-fpr", required_argument, 0,  THREAT_SENSOR_FPR },
    {"threat-sensor-fnr",  required_argument, 0,  THREAT_SENSOR_FNR },
    {"target-sensor-fpr", required_argument, 0,  TARGET_SENSOR_FPR },
    {"target-sensor-fnr",  required_argument, 0,  TARGET_SENSOR_FNR },
	{"dl-target-sensor-range", required_argument, 0,  DL_TARGET_SENSOR_RANGE },
	{"auto-range", no_argument, 0,  AUTO_RANGE },
    {"decision-horizon",  required_argument, 0,  DECISION_HORIZON },
	{"observation-horizon", required_argument, 0, OBSERVATION_HORIZON},
	{"accumulate-observations", no_argument, 0, ACCUMULATE_OBSERVATIONS},
    {"reach-path",  required_argument, 0,  REACH_PATH },
    {"reach-model",  required_argument, 0,  REACH_MODEL },
	{"distrib-approx", required_argument, 0, DISTRIB_APPROX },
	{"change-alt-latency", required_argument, 0, CHANGE_ALT_LATENCY },
	{"non-latency-aware", no_argument, 0, NON_LATENCY_AWARE },
	{"seed", required_argument, 0, SEED },
	{"probability-bound", required_argument, 0, PROBABILITY_BOUND },
	{"stay-alive-reward", required_argument, 0, STAY_ALIVE_REWARD },
	{"no-formation", no_argument, 0, NO_FORMATION },
	{"ecm", no_argument, 0, ECM },
	{"two-level-tactics", no_argument, 0, TWO_LEVEL_TACTICS },
	{"adapt-mgr", required_argument, 0, ADAPT_MGR },
    {"prism-template",  required_argument, 0,  PRISM_TEMPLATE },
	{"opt-test", no_argument, 0, OPT_TEST },
	{"hp-mode", required_argument, 0, HP_MODE },
	{"ecm-threat", required_argument, 0, ECM_THREAT_PROB },
	{"ecm-target", required_argument, 0, ECM_TARGET_PROB },
	{"error-tolerance", required_argument, 0, ERROR_TOLERANCE },
	{"detection-formation-factor", required_argument, 0, DETECTION_FORMATION_FACTOR },
	{"destruction-formation-factor", required_argument, 0, DESTRUCTION_FORMATION_FACTOR },
#if DART_USE_CE
	{"ce-nonincremental", no_argument, 0, CE_NONINCREMENTAL },
	{"ce-hint-weight", required_argument, 0, CE_HINT_WEIGHT },
	{"ce-samples", required_argument, 0, CE_SAMPLES },
	{"ce-alpha", required_argument, 0, CE_ALPHA },
	{"ce-precision", required_argument, 0, CE_PRECISION },
	{"ce-max-iterations", required_argument, 0, CE_MAX_ITERATIONS },
#endif
    {0, 0, 0, 0 }
};

void usage() {
	cout << "valid options are:" << endl;
	int opt = 0;
	while (long_options[opt].name != 0) {
		cout << "\t--" << long_options[opt].name;
		if (long_options[opt].has_arg == required_argument) {
			cout << "=value";
		}
		cout << endl;
		opt++;
	}
	exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
	SimulationParams simParams;
	Params adaptParams;
	bool autoRange = false;
    int seed = -1;

	while (true) {
		int option_index = 0;

		auto c = getopt_long(argc, argv, "", long_options, &option_index);

		if (c == -1) {
			break;
		}

		switch (c) {
		case MAP_SIZE:
			simParams.scenario.MAP_SIZE = atoi(optarg);
			break;
		case SQUARE_MAP:
			simParams.scenario.SQUARE_MAP = true;
			break;
		case NUM_TARGETS:
			simParams.scenario.TARGETS = atoi(optarg);
			break;
		case NUM_THREATS:
			simParams.scenario.THREATS = atoi(optarg);
			break;
		case ALTITUDE_LEVELS:
			adaptParams.configurationSpace.ALTITUDE_LEVELS = atoi(optarg);
			break;
		case THREAT_RANGE:
			adaptParams.environmentModel.THREAT_RANGE = atoi(optarg);
			break;
		case THREAT_SENSOR_FPR:
			adaptParams.longRangeSensor.THREAT_SENSOR_FPR = atof(optarg);
			break;
		case THREAT_SENSOR_FNR:
			adaptParams.longRangeSensor.THREAT_SENSOR_FNR = atof(optarg);
			break;
		case TARGET_SENSOR_FPR:
			adaptParams.longRangeSensor.TARGET_SENSOR_FPR = atof(optarg);
			break;
		case TARGET_SENSOR_FNR:
			adaptParams.longRangeSensor.TARGET_SENSOR_FNR = atof(optarg);
			break;
		case DL_TARGET_SENSOR_RANGE:
			adaptParams.environmentModel.TARGET_SENSOR_RANGE = atoi(optarg);
			break;
		case AUTO_RANGE:
			autoRange = true;
			break;
		case DECISION_HORIZON:
			adaptParams.adaptationManager.HORIZON = atoi(optarg);
			break;
		case OBSERVATION_HORIZON:
			adaptParams.longRangeSensor.OBSERVATION_HORIZON = atoi(optarg);
			break;
		case ACCUMULATE_OBSERVATIONS:
			adaptParams.adaptationManager.accumulateObservations = true;
			break;
		case REACH_MODEL:
			adaptParams.adaptationManager.REACH_MODEL = optarg;
			break;
		case REACH_PATH:
			adaptParams.adaptationManager.REACH_PATH = optarg;
			break;
		case DISTRIB_APPROX:
			adaptParams.adaptationManager.distributionApproximation =
					(DartDTMCEnvironment::DistributionApproximation) atoi(optarg);
			break;
		case CHANGE_ALT_LATENCY:
			adaptParams.tactics.changeAltitudeLatency = atof(optarg);
			break;
		case NON_LATENCY_AWARE:
			adaptParams.adaptationManager.nonLatencyAware = true;
			break;
		case SEED:
            seed = atoi(optarg);
			RandomSeed::seed(seed);
			break;
		case PROBABILITY_BOUND:
			adaptParams.adaptationManager.probabilityBound = atof(optarg);
			break;
		case STAY_ALIVE_REWARD:
			adaptParams.adaptationManager.finalReward = atof(optarg);
			break;
		case NO_FORMATION:
			adaptParams.adaptationManager.REACH_MODEL += "-formation-disabled";
			break;
		case ECM:
			adaptParams.configurationSpace.hasEcm = true;
			break;
		case TWO_LEVEL_TACTICS:
			adaptParams.adaptationManager.twoLevelTactics = true;
			break;
		case ADAPT_MGR:
			adaptParams.adaptationManager.mgr = optarg;
			break;
		case PRISM_TEMPLATE:
			adaptParams.adaptationManager.PRISM_TEMPLATE = optarg;
			break;
		case OPT_TEST:
			simParams.optimalityTest = true;
			break;
		case HP_MODE:
			adaptParams.adaptationManager.hpMode = optarg;
			break;
		case ECM_THREAT_PROB:
			adaptParams.longRangeSensor.THREAT_ECM_PROBABILITY= atof(optarg);
			break;
		case ECM_TARGET_PROB:
			adaptParams.longRangeSensor.TARGET_ECM_PROBABILITY= atof(optarg);
			break;
        case ERROR_TOLERANCE:
            adaptParams.longRangeSensor.ERROR_TOLERANCE = atof(optarg);
            break;
        case DETECTION_FORMATION_FACTOR:
            adaptParams.environmentModel.TARGET_DETECTION_FORMATION_FACTOR = atof(optarg);
            break;
        case DESTRUCTION_FORMATION_FACTOR:
            adaptParams.environmentModel.DESTRUCTION_FORMATION_FACTOR = atof(optarg);
            break;
#if DART_USE_CE
		case CE_NONINCREMENTAL:
			adaptParams.adaptationManager.ce_incremental = false;
			break;
		case CE_HINT_WEIGHT:
			adaptParams.adaptationManager.ce_hintWeight = atof(optarg);
			break;
		case CE_SAMPLES:
			adaptParams.adaptationManager.ce_samples = atoi(optarg);
			break;
		case CE_ALPHA:
			adaptParams.adaptationManager.ce_alpha = atof(optarg);
			break;
		case CE_PRECISION:
			adaptParams.adaptationManager.ce_precision = atof(optarg);
			break;
		case CE_MAX_ITERATIONS:
			adaptParams.adaptationManager.ce_maxIterations = atoi(optarg);
			break;
#endif
		default:
			usage();
		}
	}

	if (optind < argc) {
		usage();
	}

	// Handle the observation horizon cases
	if(adaptParams.longRangeSensor.OBSERVATION_HORIZON == 0){
		adaptParams.longRangeSensor.OBSERVATION_HORIZON = adaptParams.adaptationManager.HORIZON;
	} else if(adaptParams.longRangeSensor.OBSERVATION_HORIZON < adaptParams.adaptationManager.HORIZON){
		cout << "Observation horizon must be greater than or equal to the decision horizon." << endl;
		return 0;
	}

	if (adaptParams.adaptationManager.mgr == "hybrid"
			&& (adaptParams.adaptationManager.hpMode != "ml0"
					&& adaptParams.adaptationManager.hpMode != "ml1"
					&& adaptParams.adaptationManager.hpMode != "pg"
					&& adaptParams.adaptationManager.hpMode != "so"
					&& adaptParams.adaptationManager.hpMode != "si"
					&& adaptParams.adaptationManager.hpMode != "cb1"
					&& adaptParams.adaptationManager.hpMode != "cb2"
					&& adaptParams.adaptationManager.hpMode != "cb3")) {
		cout << "ERROR: If adapt-mgr is hybrid, hp-mode (e.g., ml0, ml1, cb1, cb2, cb3, pg, so, si) is required" << endl;
		return 0;
	}

    if (adaptParams.adaptationManager.mgr == "pmc") {
        adaptParams.adaptationManager.hpMode = "fast";
    }

	if(adaptParams.adaptationManager.accumulateObservations && simParams.scenario.SQUARE_MAP){
		cout << "'--accumulate-observations' and '--square-map' are not compatible." << endl;
		return 0;
	}

	if (autoRange) {
		adaptParams.environmentModel.TARGET_SENSOR_RANGE = adaptParams.configurationSpace.ALTITUDE_LEVELS;
		adaptParams.environmentModel.THREAT_RANGE = adaptParams.configurationSpace.ALTITUDE_LEVELS * 3 / 4;
	}

	if (adaptParams.adaptationManager.twoLevelTactics) {
		adaptParams.adaptationManager.REACH_MODEL += "-2l";
	}

	if (adaptParams.configurationSpace.hasEcm) {
		adaptParams.adaptationManager.REACH_MODEL += "-ecm";
	}

    // TODO also write command line
    string parentDir = "/home/ashutosp/dart/logs";
    DebugFileInfo::getInstance(seed, parentDir.c_str(), (adaptParams.adaptationManager.hpMode).c_str());
    
    for(int i = 0; i < argc; ++i) {
        DebugFileInfo::getInstance()->write(argv[i], false);
        DebugFileInfo::getInstance()->write(" ", false);
    }
    DebugFileInfo::getInstance()->writeEndLine();

	// generate environment
#if FIXED2DSPACE
	RealEnvironment threatEnv;
	threatEnv.populate(Coordinate(10, 10), 0);

	RealEnvironment targetEnv;
	targetEnv.populate(Coordinate(10, 10), 0);

	threatEnv.setAt(Coordinate(2,2), true);
	threatEnv.setAt(Coordinate(3,2), true);
	threatEnv.setAt(Coordinate(6,6), true);
	threatEnv.setAt(Coordinate(7,7), true);

	targetEnv.setAt(Coordinate(5,2), true);
	targetEnv.setAt(Coordinate(7,2), true);
	targetEnv.setAt(Coordinate(7,5), true);
#else
	RealEnvironment threatEnv;
	RealEnvironment targetEnv;

	if (simParams.scenario.SQUARE_MAP) {

		/* generate true environment */
		threatEnv.populate(Coordinate(simParams.scenario.MAP_SIZE, simParams.scenario.MAP_SIZE), simParams.scenario.THREATS);
		targetEnv.populate(Coordinate(simParams.scenario.MAP_SIZE, simParams.scenario.MAP_SIZE), simParams.scenario.TARGETS);
	} else {

		/* generate true environment */
		threatEnv.populate(Coordinate(simParams.scenario.MAP_SIZE, 1), simParams.scenario.THREATS);
		targetEnv.populate(Coordinate(simParams.scenario.MAP_SIZE, 1), simParams.scenario.TARGETS);
	}
#endif


	// generate route
	Route route;

#if FIXED2DSPACE
	unsigned x = 2;
	unsigned y = 2;
	while (x < 7) {
		route.push_back(Coordinate(x,y));
		x++;
	}
	while (y <= 6) {
		route.push_back(Coordinate(x,y));
		y++;
	}
#else
	if (simParams.scenario.SQUARE_MAP) {
		for (unsigned y = 0; y < simParams.scenario.MAP_SIZE; y++) {
			if (y % 2) {
				for (unsigned x = simParams.scenario.MAP_SIZE; x > 0; x--) {
					route.push_back(Coordinate(x - 1, y));
				}
			} else {
				for (unsigned x = 0; x < simParams.scenario.MAP_SIZE; x++) {
					route.push_back(Coordinate(x, y));
				}
			}
		}
	} else {
		route = Route(Coordinate(0,0), 1.0, 0.0, simParams.scenario.MAP_SIZE);
	}
#endif

	// change parameters if doing optimality test
	if (simParams.optimalityTest) {
		adaptParams.adaptationManager.HORIZON = route.size();
		adaptParams.longRangeSensor.TARGET_SENSOR_FNR = 0;
		adaptParams.longRangeSensor.TARGET_SENSOR_FPR = 0;
		adaptParams.longRangeSensor.THREAT_SENSOR_FNR = 0;
		adaptParams.longRangeSensor.THREAT_SENSOR_FPR = 0;
		adaptParams.adaptationManager.distributionApproximation = DartDTMCEnvironment::DistributionApproximation::POINT;

		// autorange
		adaptParams.environmentModel.TARGET_SENSOR_RANGE = adaptParams.configurationSpace.ALTITUDE_LEVELS / 2.0;
		adaptParams.environmentModel.THREAT_RANGE = adaptParams.configurationSpace.ALTITUDE_LEVELS * 3.0 / 4;
		cout << "ranges sensor=" << adaptParams.environmentModel.TARGET_SENSOR_RANGE
				<< " threat=" << adaptParams.environmentModel.THREAT_RANGE << endl;
	}

    //cout << "Inside main ----- adaptParams.environmentModel.THREAT_RANGE = " << adaptParams.environmentModel.THREAT_RANGE << endl;
    //cout << "Inside main ----- adaptParams.configurationSpace.ALTITUDE_LEVELS = " 
    //        << adaptParams.configurationSpace.ALTITUDE_LEVELS << endl;

	// instantiate adaptation manager
	shared_ptr<TargetSensor> pTargetSensor = Simulation::createTargetSensor(simParams,
			adaptParams);
	shared_ptr<Threat> pThreatSim = Simulation::createThreatSim(simParams, adaptParams);

	/* initialize adaptation manager */
	DartAdaptationManager adaptMgr;
	adaptMgr.initialize(adaptParams,
			unique_ptr<pladapt::UtilityFunction>(
					new DartUtilityFunction(pThreatSim, pTargetSensor,
							adaptParams.adaptationManager.finalReward)));

	if (simParams.optimalityTest && !adaptMgr.supportsStrategy()) {
		throw std::invalid_argument("selected adaptation manager does not support full strategies");
	}

	auto results = Simulation::run(simParams, adaptParams, threatEnv, targetEnv,
			route, adaptMgr);

	const std::string RESULTS_PREFIX = "out:";

	cout << RESULTS_PREFIX << "destroyed=" << results.destroyed << endl;
    DebugFileInfo::getInstance()->write(RESULTS_PREFIX + "destroyed=" + to_string(results.destroyed));

	cout << RESULTS_PREFIX << "targetsDetected=" << results.targetsDetected << endl;
    DebugFileInfo::getInstance()->write(RESULTS_PREFIX + "targetsDetected=" + to_string(results.targetsDetected));
	
    cout << RESULTS_PREFIX << "missionSuccess=" << results.missionSuccess << endl;
    DebugFileInfo::getInstance()->write(RESULTS_PREFIX + "missionSuccess=" + to_string(results.missionSuccess));

    std::string summaryFile = "summary.csv";
	cout << "csv," << results.targetsDetected << ',' << results.destroyed
			<< ',' << results.whereDestroyed.x
			<< ',' << results.missionSuccess
			<< ',' << results.decisionTimeAvg
			<< ',' << results.decisionTimeVar
			<< ',' << results.numQuickDecisions
			<<  endl;

    std::fstream summaryWriter;
    summaryWriter.open(summaryFile.c_str(), std::fstream::in | std::fstream::out | std::fstream::app);

    summaryWriter << seed << ", "
            << adaptParams.adaptationManager.mgr << ", "
            << adaptParams.adaptationManager.hpMode << ", "
            << adaptParams.adaptationManager.HORIZON << ", "
            << adaptParams.longRangeSensor.OBSERVATION_HORIZON << ", "
            << adaptParams.configurationSpace.hasEcm << ", "
            << adaptParams.configurationSpace.ALTITUDE_LEVELS << ", "
            << adaptParams.adaptationManager.accumulateObservations << ", "
            << simParams.scenario.MAP_SIZE << ", "
            << adaptParams.adaptationManager.twoLevelTactics << ", "
            << simParams.scenario.TARGETS << ", "
            << simParams.scenario.THREATS << ", "
            << adaptParams.adaptationManager.finalReward << ", "
            << adaptParams.longRangeSensor.THREAT_ECM_PROBABILITY << ", "
            << adaptParams.longRangeSensor.TARGET_ECM_PROBABILITY << ", "
            << adaptParams.longRangeSensor.ERROR_TOLERANCE << ", "
            << adaptParams.environmentModel.TARGET_DETECTION_FORMATION_FACTOR << ", "
            << adaptParams.environmentModel.DESTRUCTION_FORMATION_FACTOR << ", "
            << results.targetsDetected << ", " 
            << results.destroyed << ", " 
            << results.whereDestroyed.x << ", " 
            << results.missionSuccess << ", " 
            << results.decisionTimeAvg << ", " 
            << results.decisionTimeVar << ", " 
            << results.numQuickDecisions <<  ", "
            << results.deliberativePlanFailedCount << ", "
            << results.reactivePlanningCount << endl;
            
    summaryWriter.close();

	return 0;
}
