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
#include <pladapt/HybridAdaptationManager.h>
#include <pladapt/Utils.h>
#include <pladapt/PMCRAAdaptationManager.h>
#include "../examples/dart/dartam/include/dartam/DebugFileInfo.h"
#include <iostream>
#include <sstream>
#include <pladapt/State.h>
#include <assert.h>
#include <boost/filesystem.hpp>
#include <pladapt/ClassifyProblemInstance.h>

using namespace std;

namespace dart {
namespace am2 {
const char* HybridAdaptationManager::NO_LATENCY = "nolatency";
const char* HybridAdaptationManager::TEMPLATE_PATH = "templatePath";
const std::string PCTL = "Rmax=? [ F \"final\" ]";

HybridAdaptationManager::HybridAdaptationManager(const string& mode) : savedDTMC(0),
		pathToStoreProfilingProblems("/home/ashutosp/ProfilingProblems/") {
    if (mode == "pg") hpMode = HpMode::PG;
    else if (mode == "cb") hpMode = HpMode::CB;
    else if (mode == "ml0") hpMode = HpMode::ML0;
    else if (mode == "ml1") hpMode = HpMode::ML1;
    else if (mode == "so") hpMode = HpMode::SLOWONLY;
    else if (mode == "si") hpMode = HpMode::SLOWINSTANT;
    else assert(false);

    if (hpMode == HpMode::ML0 || hpMode == HpMode::ML1) {
        ClassifyProblemInstance::getInstance("/home/ashutosp/dart/examples/dart/PythonSrc/consolidated-db.csv", "");
    }
}

void HybridAdaptationManager::initialize(std::shared_ptr<const pladapt::ConfigurationManager> configMgr,
                                            const YAML::Node& params,
                                            std::shared_ptr<const DartPMCHelper> helper) {
	pConfigMgr = configMgr;
	this->params = params;
	pMcHelper = helper;
    fastPlanPath = "";
    slowPlanPath = "";
    deliberativeFailedCount = 0;
    reactivePlanningCount = 0;
}

pladapt::TacticList HybridAdaptationManager::evaluate(const pladapt::Configuration& currentConfigObj,
                                                        const pladapt::EnvironmentDTMCPartitioned& envDTMC,
                                                        const pladapt::UtilityFunction& utilityFunction,
                                                        unsigned horizon, double destroyProbability,
                                                        double detectionProbability) {

	// QUESTION: Is it possible for the model to be open at this point but not #drew
	//  be loaded into the PlanDB? If so it needs to be accounted for here

    unsigned adjustedTimestep = (dynamic_cast<const DartConfiguration&>(currentConfigObj)).getTimestep() - planStartTime;
    DartConfiguration adjustedConfig = DartConfiguration(dynamic_cast<const DartConfiguration&>(currentConfigObj));
    adjustedConfig.setTimestep(adjustedTimestep);
    pladapt::TacticList tactics;
    static unsigned reused = 0;
    double classifierLabel = -1.0;

	// Check PlanDB for a the existance of the current state
	State currentState;
	PlanDB::get_instance()->populate_state_obj(&adjustedConfig, 
            &savedDTMC, &envDTMC, currentState,
            dynamic_cast<const DartPMCHelper&>(*pMcHelper).errorTolerance);

	// If there is no applicable plan exists generate a new one
	if ((currentState.env_state == UINT_MAX) || (adjustedTimestep >= horizon)) {
        if (currentState.env_state == UINT_MAX) {
            ++deliberativeFailedCount;
            DebugFileInfo::getInstance()->write("Deliberative Plan Failed");
            cout << "Deliberative Plan Failed" << endl;
        } else {
            DebugFileInfo::getInstance()->write("Deliberative Plan Over");
            cout << "Deliberative Plan Over" << endl;
        }

        reused = 0;
		planStartTime = (dynamic_cast<const DartConfiguration&>(currentConfigObj)).getTimestep();
		PlanDB::get_instance()->clean_db();

		/* check if we need to adjust the horizon to the environment size */
		if ((envDTMC.getNumberOfParts() - 1) < horizon) {
			if (envDTMC.getNumberOfParts() > 1 && envDTMC.isLastPartFinal()) {
				horizon = envDTMC.getNumberOfParts() - 1;
				cout << "warning: environment is shorter than horizon" << endl;
			}
		}

		// Generate PRISM initialization strings
		const string initialState = pMcHelper->generateInitializations(currentConfigObj, utilityFunction, horizon);

		std::vector<double> targetPredictions;
		std::vector<double> threatPredictions;

        string environmentModel = generateEnvironmentDTMC(envDTMC, targetPredictions, threatPredictions);

        std::vector<double>::iterator itr1 = targetPredictions.begin();
        std::vector<double>::iterator itr2 = threatPredictions.begin();

        /*while (itr1 != targetPredictions.end()) {
            cout << i << " Target = " << *itr1 << "    Threat = " << *itr2 << endl;
            ++itr1; ++itr2; ++i;
        }*/


        if (hpMode == PG) {
            environmentModel += string("// #ENV ENDS\n");            
        }

		string templatePath = params[TEMPLATE_PATH].as<string>();
		if (params[NO_LATENCY].as<bool>()) {
			templatePath += "-nl";
		}

		templatePath += ".prism";
		deliberativeWrapper.setModelTemplatePath(templatePath);

		// Generates the prism model and adversary transition model
        DebugFileInfo::getInstance()->write("Slow Planning Triggered");
		deliberativeWrapper.generatePersistentPlan(environmentModel, initialState, PCTL);
		slowPlanPath = deliberativeWrapper.getModelDirectory();
		DebugFileInfo::getInstance()->write("slowPlanPath = " + slowPlanPath);
		cout << "slowPlanPath = " << slowPlanPath << endl;

		// Load the plan into PlanDB
		PlanDB::get_instance()->populate_db(slowPlanPath.c_str());

		savedDTMC = envDTMC;

		//TODO: Add switch statement to easily change between the various hybrid planners #drew
        // There is no straightforward way to pass the information to this point from
        // command line params 

        // Check threat level
        cout << "Threat range:" << dynamic_cast<const DartPMCHelper&>(*pMcHelper).threatRange << endl;
        cout << "Destroy Probability = " << destroyProbability << endl;
        cout << "Detection Probability = " << detectionProbability << endl;
        cout << "Altitude level = " << adjustedConfig.getAltitudeLevel() << endl;

        DebugFileInfo::getInstance()
                ->write("Threat range:" + std::to_string(dynamic_cast<const DartPMCHelper&>(*pMcHelper).threatRange));

        if (hpMode != PG && (hpMode == HpMode::ML0 || hpMode == HpMode::ML1)) {
            classifierLabel  = ClassifyProblemInstance::getInstance()->useReactive(&adjustedConfig,
                                            targetPredictions, threatPredictions);
        }

        if (hpMode == PG 
                || (hpMode == CB && destroyProbability >= 0.6)
                || (hpMode == HpMode::ML0 && classifierLabel != 1.0)
                || (hpMode == HpMode::ML1 && classifierLabel != 0.0)) {
                //|| adjustedConfig.getAltitudeLevel() < dynamic_cast<const DartPMCHelper&>(*pMcHelper).threatRange) {
            if (hpMode == CB) {
                DebugFileInfo::getInstance()->write("In danger: ");
                cout << "In danger: ";
                DebugFileInfo::getInstance()->write("Destroy Probability = " + to_string(destroyProbability));
                DebugFileInfo::getInstance()->write("Detection Probability = " + to_string(detectionProbability));
            }

            cout << "Fast Planning Triggered" << endl;
            ++reactivePlanningCount;
            DebugFileInfo::getInstance()->write("Fast Planning Triggered");

            auto pAdaptMgr = pladapt::PMCAdaptationManager();
            pAdaptMgr.initialize(pConfigMgr, params, pMcHelper);

            //cout << "Assert-1 HybridAdaptationManager::evaluate" << endl;
            //assert(false);

            unsigned reactiveHorizon = 2;
            tactics = pAdaptMgr.evaluate(currentConfigObj, envDTMC, utilityFunction, reactiveHorizon);
            fastPlanPath = pAdaptMgr.getPlanPath();
		    DebugFileInfo::getInstance()->write("fastPlanPath = " + fastPlanPath);
            cout << "fastPlanPath = " << fastPlanPath << endl;
            //cout << "Assert-2 HybridAdaptationManager::evaluate" << endl;
            //assert(false);
            
            if (hpMode == PG) {
                int seed = DebugFileInfo::getInstance()->getSimulationSeed();

                DumpPlanningProblems::get_instance(pathToStoreProfilingProblems, seed)
                                    ->copySampleProblems(fastPlanPath, slowPlanPath, &adjustedConfig,
                                            targetPredictions, threatPredictions, classifierLabel);
            }
        } else {
            cout << "Safe: Waiting for plan" << endl;
            DebugFileInfo::getInstance()->write("Safe: Waiting for deliberative plan");
        }
	} else { // If there is an applicable plan, use it

		// Use the plan
		PlanDB::Plan p;
		PlanDB::get_instance()->get_plan(&adjustedConfig, &savedDTMC, &envDTMC,
                p, dynamic_cast<const DartPMCHelper&>(*pMcHelper).errorTolerance);
        DebugFileInfo::getInstance()->write("Slow Plan Reused = " + to_string(++reused));
		

        PlanDB::Plan::iterator itr = p.begin();
        while (itr != p.end()) {
            //cout << "HybridAdaptationManager::evaluate plan = " << *itr << endl;
            tactics.insert(*itr);
            ++itr;
        }

        // Convert the vector of strings into a set of strings to remain compatable
		//  with Gabe's existing code
		//tactics(p.begin(), p.end());
	}

    pladapt::TacticList::iterator it = tactics.begin();

    while (it != tactics.end()) {
        DebugFileInfo::getInstance()->write(*it + " ", false);
        //cout << "HybridAdaptationManager::evaluate ##### tactic = " << *it << endl;
        ++it;
    }
    
    if (tactics.size() == 0) {
        DebugFileInfo::getInstance()->write("No Tactic Suggested");
    } else {
        DebugFileInfo::getInstance()->writeEndLine();
    }

    return tactics;
}

void HybridAdaptationManager::cleanupModel() const {
    //cout << "deleting slowPath = " << slowPlanPath << endl;
    //cout << "deleting fastPath = " << fastPlanPath << endl;

    boost::filesystem::path slow(slowPlanPath);
    boost::filesystem::path fast(fastPlanPath);
    
    if (boost::filesystem::exists(slow)) {
        boost::filesystem::remove_all(slow);
    }

    if (boost::filesystem::exists(fast)) {
        boost::filesystem::remove_all(fast);
    }
}

// these constants depend on the PRISM model
const string STATE_VAR = "s";
const string GUARD = "[tick] ";

std::string HybridAdaptationManager::generateEnvironmentDTMC(const pladapt::EnvironmentDTMCPartitioned& envDTMC,
        std::vector<double>& targetPredictions, std::vector<double>& threatPredictions) {
	const string STATE_VALUE_FORMULA = "formula stateValue";

	string result;

	// generate state value formulas
	const int numComponents = envDTMC.getStateValue(0).getNumberOfComponents();
	stringstream stateValueFormulas[numComponents];
	for (int c = 0; c < numComponents; c++) {
		stateValueFormulas[c] << STATE_VALUE_FORMULA;
		if (c > 0) {
			stateValueFormulas[c] << c;
		}
        //cout << "##### " << (stateValueFormulas[c]).str() << endl;

		stateValueFormulas[c] << " = ";
	}

	const string padding(stateValueFormulas[0].str().length(), ' ');
	for (unsigned s = 0; s < envDTMC.getNumberOfStates(); s++) {
		const auto& envValue = envDTMC.getStateValue(s);
		for (int c = 0; c < numComponents; c++) {
			if (s > 0) {
				stateValueFormulas[c] << " + " << endl << padding;
			}
			stateValueFormulas[c] << "(" << STATE_VAR << " = " << s << " ? "
			                      << envValue.getComponent(c).asDouble() << " : 0)";

            if (c == 0) {
                // Threats
                threatPredictions.push_back(envValue.getComponent(c).asDouble());
            } else if (c == 1) {
                // Targets
                targetPredictions.push_back(envValue.getComponent(c).asDouble());
            } else {
                assert(false);
            }
		}
	}

	for (int c = 0; c < numComponents; c++) {
		stateValueFormulas[c] << ';' << endl;
	}

	// generate state transitions
	stringstream out;

	out << STATE_VAR << " : [0.." << envDTMC.getNumberOfStates() - 1 << "] init 0;" << endl;

	for (unsigned from = 0; from < envDTMC.getNumberOfStates(); from++) {
		bool firstTransition = true;
		for (unsigned to = 0; to < envDTMC.getNumberOfStates(); to++) {
			double probability = envDTMC.getTransitionMatrix() (from, to);
			if (probability > 0.0) {
				if (firstTransition) {
					firstTransition = false;
					out << GUARD << STATE_VAR << " = "
					    << from << " -> " << endl;
					out << '\t';
				} else {
					out << endl << "\t+ ";
				}
				out << probability << " : (" << STATE_VAR << "' = " << to
				    << ")";
			}
		}
		if (!firstTransition) {
			out << ';' << endl;
		}
	}

    assert(targetPredictions.size() == threatPredictions.size() && threatPredictions.size() == 46);
    double arr[] = {    0.0,
                        0.034225, 0.11655, 0.034225, 0.11655, 0.3969, 0.11655, 0.034225, 0.11655, 0.034225,
                        0.034225, 0.11655, 0.034225, 0.11655, 0.3969, 0.11655, 0.034225, 0.11655, 0.034225,        
                        0.034225, 0.11655, 0.034225, 0.11655, 0.3969, 0.11655, 0.034225, 0.11655, 0.034225,        
                        0.034225, 0.11655, 0.034225, 0.11655, 0.3969, 0.11655, 0.034225, 0.11655, 0.034225,        
                        0.034225, 0.11655, 0.034225, 0.11655, 0.3969, 0.11655, 0.034225, 0.11655, 0.034225,        
                    };

    std::vector<double>::iterator itr1 = targetPredictions.begin();
    std::vector<double>::iterator itr2 = threatPredictions.begin();
    unsigned index = 0;


    while (itr1 != targetPredictions.end()) {
        *itr1 = (*itr1) * (arr[index]);
        *itr2 = (*itr2) * (arr[index]);
        ++itr1;
        ++itr2;
        ++index;
    }

    assert(index == 46);

	out << "endmodule" << endl;

	// append all the state value formulas
	out << endl << "// environment has " << numComponents << " components" << endl;
	for (int c = 0; c < numComponents; c++) {
		out << endl << stateValueFormulas[c].str();
	}

	return out.str();
}

HybridAdaptationManager::~HybridAdaptationManager(){

}

}   /* am2 */
} /* dart */
