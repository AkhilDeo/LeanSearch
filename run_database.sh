#!/bin/bash
#SBATCH --job-name=postgres_database
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=database_%j.log
#SBATCH --error=database_%j.err

# Export environment variables from .env
set -a
source .env
set +a

# Start PostgreSQL server in the background
postgres -D /home/adeo1/scratch/polya/src/retriever/pgdata &
POSTGRES_PID=$!

# Wait for PostgreSQL to be ready
sleep 5

# Run the database command
python -m database jixia /home/adeo1/scratch/polya/src/retriever/Mathlib Mathlib

# Kill PostgreSQL when done
kill $POSTGRES_PID
