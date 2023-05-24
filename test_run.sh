#!/bin/bash
# Parameterization settings. These will be explained in 2.2. Now simply copy them to run the test.

export CARLA_ROOT=/home/frank/code/GP_test1/carla
export SCENARIO_RUNNER_ROOT=/home/frank/code/GP_test1/scenario_runner
export LEADERBOARD_ROOT=/home/frank/code/GP_test1/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":"${LEADERBOARD_ROOT}/team_code":${PYTHONPATH}

export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=${LEADERBOARD_ROOT}/data/routes_devtest.xml
export REPETITIONS=1
export DEBUG_CHALLENGE=1
# export TEAM_AGENT=${LEADERBOARD_ROOT}/leaderboard/autoagents/human_agent.py
export TEAM_AGENT=/home/frank/code/GP_test1/leaderboard/expert_code/auto_pilot.py
export TEAM_CONFIG=/home/frank/code/GP_test1/leaderboard/expert_code/expert.yaml
export CHECKPOINT_ENDPOINT=${LEADERBOARD_ROOT}/results.json
export CHALLENGE_TRACK_CODENAME=MAP

export HOST=localhost

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--host=${HOST}  \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}
