#!/bin/bash

export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":${PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=20000 # same as the carla server port
export TM_PORT=25000 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=leaderboard/data/additional_routes/routes_town01_long.xml
#export ROUTES=leaderboard/data/training_routes/routes_town02_long.xml
export TEAM_AGENT=leaderboard/team_code/LCDiff_agent.py # agent
export TEAM_CONFIG=leaderboard/team_code/LCDiff_config.py # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=results/sample_result.json # results file
#export SCENARIOS=leaderboard/data/scenarios/no_scenarios.json
export SCENARIOS=leaderboard/data/scenarios/town01_all_scenarios.json 
export SAVE_PATH=data/expert # path for saving episodes while evaluating
export RESUME=False

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--host=192.168.3.25 \
--trafficManagerPort=${TM_PORT}

