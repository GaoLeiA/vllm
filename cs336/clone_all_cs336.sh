#!/bin/bash

# Bash script to sync CS336 repositories from GaoLeiA

REPOS=(
    "spring2025-lectures"
    "assignment1-basics"
    "assignment2-systems"
    "assignment3-scaling"
    "assignment4-data"
    "assignment5-alignment"
)

BASE_URL="https://github.com/GaoLeiA"

echo -e "\033[1;36mStarting CS336 Repository Sync...\033[0m"

for repo in "${REPOS[@]}"; do
    echo -e "\n\033[1;33mChecking $repo...\033[0m"
    
    if [ -d "$repo" ]; then
        echo "  Directory exists. Pulling latest changes..."
        cd "$repo" || exit
        git pull
        cd ..
    else
        echo "  Directory not found. Cloning..."
        url="$BASE_URL/$repo.git"
        git clone "$url"
    fi
done

echo -e "\n\033[1;32mAll operations completed.\033[0m"
