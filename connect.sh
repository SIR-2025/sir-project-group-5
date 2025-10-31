#!/usr/bin/env bash

# 1. Open a Terminal window to run redis-server
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$(pwd)'; echo 'Starting redis-server...'; redis-server conf/redis/redis.conf; exec bash"
end tell
EOF

# 2. Open another Terminal window to activate the environment and run Dialogflow CX
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$(pwd)'; source .venv/bin/activate; echo 'Environment activated'; run-dialogflow-cx; exec bash"
end tell
EOF

# 2. Open another Terminal window to run python scripts
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$(pwd)'; source .venv/bin/activate; clear; echo 'Run python scripts in this terminal'"
end tell
EOF

chmod +x connect.sh