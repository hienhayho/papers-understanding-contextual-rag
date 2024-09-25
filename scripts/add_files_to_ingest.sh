#!/bin/bash

# TYPE must be in ["origin", "contextual", "both"]
TYPE=$1

python src/ingest/add_papers.py --type "$TYPE" --files "${@:2}"
