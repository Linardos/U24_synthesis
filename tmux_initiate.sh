#!/bin/bash

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: bash $0 <session_number>"
  exit 1
fi

SESSION_NUM=$1
SESSION="U24_session_$SESSION_NUM"
ENV_NAME="U24"

# Start session with first pane
tmux new-session -d -s $SESSION -n main
tmux send-keys -t $SESSION:0.0 "conda activate $ENV_NAME && python main.py 42" C-m

# Pane 1
tmux split-window -v -t $SESSION:0.0
tmux send-keys -t $SESSION:0.1 "conda activate $ENV_NAME && python main.py 43" C-m

# Pane 2
tmux split-window -v -t $SESSION:0.1
tmux send-keys -t $SESSION:0.2 "conda activate $ENV_NAME && python main.py 44" C-m

# Pane 3
tmux split-window -v -t $SESSION:0.2
tmux send-keys -t $SESSION:0.3 "conda activate $ENV_NAME && python main.py 45" C-m

# Pane 4
tmux split-window -v -t $SESSION:0.3
tmux send-keys -t $SESSION:0.4 "conda activate $ENV_NAME && python main.py 46" C-m

# Apply even-vertical layout
tmux select-layout -t $SESSION:0 even-vertical

# Focus back on first pane and attach
tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION
