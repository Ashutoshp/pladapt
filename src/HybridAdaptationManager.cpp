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
}

void HybridAdaptationManager::initialize(std::shared_ptr<const pladapt::ConfigurationManager> configMgr,
                                            const YAML::Node& params,
                                            std::shared_ptr<const DartPMCHelper> helper) {
	pConfigMgr = configMgr;
	this->params = params;
	pMcHelper = helper;
    fastPlanPath = "";
    slowPlanPath = "";
}

pladapt::TacticList HybridAdaptationManager::evaluate(const pladapt::Configuration& currentConfigObj,
                                                        const pladapt::EnvironmentDTMCPartitioned& envDTMC,
                                                        const pladapt::UtilityFunction& utilityFunction,
                                                        unsigned horizon) {

	// QUESTION: Is it possible for the model to be open at this point but not #drew
	//  be loaded into the PlanDB? If so it needs to be accounted for here

    unsigned adjustedTimestep = (dynamic_cast<const DartConfiguration&>(currentConfigObj)).getTimestep() - planStartTime;
    DartConfiguration adjustedConfig = DartConfiguration(dynamic_cast<const DartConfiguration&>(currentConfigObj));
    adjustedConfig.setTimestep(adjustedTimestep);
    pladapt::TacticList tactics;

	// Check PlanDB for a the existance of the current state
	State currentState;
	PlanDB::get_instance()->populate_state_obj(&adjustedConfig, &savedDTMC, &envDTMC, currentState);

	// If there is no applicable plan exists generate a new one
	if ((currentState.env_state == UINT_MAX) || (adjustedTimestep >= horizon)) {
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
		const string environmentModel = generateEnvironmentDTMC(envDTMC);

		string templatePath = params[TEMPLATE_PATH].as<string>();
		if (params[NO_LATENCY].as<bool>()) {
			templatePath += "-nl";
		}

		templatePath += ".prism";
		deliberativeWrapper.setModelTemplatePath(templatePath);

		// Generates the prism model and adversary transition model
		deliberativeWrapper.generatePersistentPlan(environmentModel, initialState, PCTL);
		slowPlanPath = deliberativeWrapper.getModelDirectory();
		//cout << "slowPlanPath = " << slowPlanPath << endl;

		// Load the plan into PlanDB
		PlanDB::get_instance()->populate_db(slowPlanPath.c_str());

		savedDTMC = envDTMC;

		//TODO: Add switch statement to easily change between the various hybrid planners #drew
        // There is no straightforward way to pass the information to this point from
        // command line params 

		double classifierLabel = -1;
		string envModel = "";

        // Check threat level
        cout << "Threat range:" << dynamic_cast<const DartPMCHelper&>(*pMcHelper).threatRange << endl;
        if (hpMode == PG 
                || (hpMode == CB
                		&& adjustedConfig.getAltitudeLevel() < dynamic_cast<const DartPMCHelper&>(*pMcHelper).threatRange)) {
            if (hpMode == CB) {
                cout << "In danger: ";
            }

            cout << "Fast Planning Triggered" << endl;

            auto pAdaptMgr = pladapt::PMCAdaptationManager();
            pAdaptMgr.initialize(pConfigMgr, params, pMcHelper);

            unsigned reactiveHorizon = 2;
            tactics = pAdaptMgr.evaluate(currentConfigObj, envDTMC, utilityFunction, reactiveHorizon);
            fastPlanPath = pAdaptMgr.getPlanPath();
		    cout << "fastPlanPath = " << fastPlanPath << endl;
            
            if (hpMode == PG) {
                int seed = DebugFileInfo::getInstance()->getSimulationSeed();

                DumpPlanningProblems::get_instance(pathToStoreProfilingProblems, seed)
                                    ->copySampleProblems(fastPlanPath, slowPlanPath, currentState,
                                            envModel, classifierLabel);
            }
        } else {
            cout << "Safe: Waiting for plan" << endl;
        }
	} else { // If there is an applicable plan, use it

		// Use the plan
		PlanDB::Plan p;
		PlanDB::get_instance()->get_plan(&adjustedConfig, &savedDTMC, &envDTMC, p);

		// Convert the vector of strings into a set of strings to remain compatable
		//  with Gabe's existing code
		pladapt::TacticList tactics(p.begin(), p.end());
	}

    return tactics;
}

void HybridAdaptationManager::cleanupModel() const {
    boost::filesystem::path slow(slowPlanPath);
    boost::filesystem::path fast(fastPlanPath);
    
    if (boost::filesystem::exists(slow)
            && boost::filesystem::exists(fast)) {
        boost::filesystem::remove_all(slow);
        boost::filesystem::remove_all(fast);
    }
}

// these constants depend on the PRISM model
const string STATE_VAR = "s";
const string GUARD = "[tick] ";

std::string HybridAdaptationManager::generateEnvironmentDTMC(const pladapt::EnvironmentDTMCPartitioned& envDTMC) {
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
