#!/bin/bash

# Script to delete all files starting with ._ in the repository
# Excludes important directories to avoid breaking the project

echo "Cleaning up files starting with ._ in the repository..."
echo "This will exclude .git and node_modules directories for safety."

# Find and delete files starting with ._ excluding .git and node_modules
find . -name "._*" -type f ! -path "./.git/*" ! -path "./node_modules/*" -print -delete

echo "Cleanup completed!"
echo ""
echo "Note: Files in .git and node_modules directories were preserved for safety."
echo "If you want to clean those as well, run:"
echo "  find . -name '._*' -type f -print -delete" 