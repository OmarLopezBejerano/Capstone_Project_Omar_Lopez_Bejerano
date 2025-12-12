#!/bin/bash

# Script to add, commit, and push changes to GitHub

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Adding all changes...${NC}"
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${GREEN}No changes to commit. Working tree clean.${NC}"
    exit 0
fi

# Ask for commit message
echo -e "${BLUE}Enter commit message (or press Enter for default 'Update files'):${NC}"
read -r commit_message

# Use default message if none provided
if [ -z "$commit_message" ]; then
    commit_message="Update files"
fi

echo -e "${BLUE}Committing changes...${NC}"
git commit -m "$commit_message"

echo -e "${BLUE}Pushing to GitHub...${NC}"
git push origin main

echo -e "${GREEN}Done! Changes pushed successfully.${NC}"
