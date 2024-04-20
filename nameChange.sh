#!/bin/bash

# Check if the argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

# Navigate to the directory containing the files
cd "$1" || exit

# Counter for renaming
counter=1

# Loop through each file in the directory
for file in *; do
    # Check if the item is a file (not a directory)
    if [ -f "$file" ]; then
        # Rename the file to a sequential number
        mv "$file" "$counter"
        ((counter++))
    fi
done
