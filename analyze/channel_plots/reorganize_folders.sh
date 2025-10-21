#!/bin/bash
# Reorganize gradient plot folders from layer-based to matrix-type-based structure
#
# Usage:
#   bash reorganize_folders.sh /path/to/gradient_timeseries_normalized
#   bash reorganize_folders.sh /path/to/folder --dry-run  # Preview changes

set -e  # Exit on error

# Parse arguments
INPUT_DIR="$1"
DRY_RUN=false

if [ "$2" == "--dry-run" ]; then
    DRY_RUN=true
fi

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Usage: $0 <input_directory> [--dry-run]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/gradient_timeseries_normalized"
    echo "  $0 /path/to/gradient_timeseries_normalized --dry-run"
    exit 1
fi

# Check if directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory not found: $INPUT_DIR"
    exit 1
fi

echo "======================================================================"
echo "Reorganizing gradient folders"
echo "======================================================================"
echo "Input directory: $INPUT_DIR"
echo "Mode: $([ "$DRY_RUN" = true ] && echo "DRY RUN (no changes)" || echo "EXECUTE")"
echo "======================================================================"
echo ""

# Step 1: Find all matrix types and create folders
echo "Step 1: Creating matrix type folders..."
echo "----------------------------------------------------------------------"

declare -A matrix_types

for folder in "$INPUT_DIR"/*_layer[0-9][0-9]; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    
    # Extract matrix type (everything before _layerXX)
    if [[ $folder_name =~ ^(.+)_layer[0-9]{2}$ ]]; then
        matrix_type="${BASH_REMATCH[1]}"
        matrix_types["$matrix_type"]=1
        
        if [ "$DRY_RUN" = false ]; then
            mkdir -p "$INPUT_DIR/$matrix_type"
        fi
        echo "  Found: $matrix_type"
    fi
done

echo ""
echo "Matrix types: ${!matrix_types[@]}"
echo ""

# Step 2: Copy *_all.png files to matrix type folders
echo "Step 2: Copying *_all.png files..."
echo "----------------------------------------------------------------------"

for folder in "$INPUT_DIR"/*_layer[0-9][0-9]; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    
    # Extract matrix type
    if [[ $folder_name =~ ^(.+)_layer[0-9]{2}$ ]]; then
        matrix_type="${BASH_REMATCH[1]}"
        
        # Find all *_all.png files
        for png_file in "$folder"/*_all.png; do
            if [ -f "$png_file" ]; then
                png_name=$(basename "$png_file")
                dest="$INPUT_DIR/$matrix_type/$png_name"
                
                if [ "$DRY_RUN" = true ]; then
                    echo "  [DRY RUN] $folder_name/$png_name -> $matrix_type/"
                else
                    cp "$png_file" "$dest"
                    echo "  [COPY] $folder_name/$png_name -> $matrix_type/"
                fi
            fi
        done
    fi
done

echo ""

# Step 3: Remove old layer folders
echo "Step 3: Removing old layer folders..."
echo "----------------------------------------------------------------------"

for folder in "$INPUT_DIR"/*_layer[0-9][0-9]; do
    if [ ! -d "$folder" ]; then
        continue
    fi
    
    folder_name=$(basename "$folder")
    
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY RUN] Remove: $folder_name/"
    else
        rm -rf "$folder"
        echo "  [DELETED] $folder_name/"
    fi
done

echo ""

# Step 4: Summary
echo "======================================================================"
echo "Summary"
echo "======================================================================"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] No changes made. Run without --dry-run to execute."
else
    echo "Reorganization complete!"
    echo ""
    echo "New structure:"
    for matrix in "${!matrix_types[@]}"; do
        if [ -d "$INPUT_DIR/$matrix" ]; then
            count=$(find "$INPUT_DIR/$matrix" -name "*_all.png" 2>/dev/null | wc -l)
            echo "  $matrix/: $count files"
        fi
    done
fi

echo ""
echo "Done!"
