#!/bin/bash

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: bash $0 <session_number>"
  exit 1
fi

SESSION_NUM=$1
SESSION="U24_session_$SESSION_NUM"
ENV_NAME="U24"

# Create new tmux session
tmux new-session -d -s $SESSION -n main
tmux send-keys -t $SESSION:0 "conda activate $ENV_NAME && python main.py 42" C-m

# Split vertically (bottom)
tmux split-window -v -t $SESSION:0
tmux send-keys -t $SESSION:0.1 "conda activate $ENV_NAME && python main.py 43" C-m

# Split top pane horizontally (left side)
tmux select-pane -t $SESSION:0.0
tmux split-window -h -t $SESSION:0
tmux send-keys -t $SESSION:0.2 "conda activate $ENV_NAME && python main.py 44" C-m

# Split bottom pane horizontally (left side)
tmux select-pane -t $SESSION:0.1
tmux split-window -h -t $SESSION:0
tmux send-keys -t $SESSION:0.3 "conda activate $ENV_NAME && python main.py 45" C-m

# Create 5th pane by splitting the bottom-right pane horizontally
tmux select-pane -t $SESSION:0.3
tmux split-window -v -t $SESSION:0
tmux send-keys -t $SESSION:0.4 "conda activate $ENV_NAME && python main.py 46" C-m

# Apply tiled layout for visibility
tmux select-layout -t $SESSION:0 tiled

# Focus on the first pane and attach
tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION
