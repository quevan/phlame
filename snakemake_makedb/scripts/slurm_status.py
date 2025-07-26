#!/usr/bin/env python3
"""
Submit this clustering script for sbatch to snakemake

CHANGELOG:
2023.04.08 Evan: Created to parse engaging slurm jobs statuses which are slightly different from c3ddb
2024.03.21 Evan: Included 'OUT_OF_ME+' as a fail state
"""

import os
import sys
import warnings
import subprocess


jobid = sys.argv[-1]
if jobid.startswith('Submitted batch job '):
    jobid = jobid.replace('Submitted batch job ','')


state = subprocess.run(['sacct','-j',jobid,'-X','--format=State'],stdout=subprocess.PIPE).stdout.decode('utf-8')
state = state.split('\n')[2].strip()

map_state={"PENDING":'running',
           "RUNNING":'running', 
           "SUSPENDED":'running', 
           "CANCELLED":'failed', 
           "CANCELLED+":'failed',
           "COMPLETING":'running', 
           "COMPLETED":'success', 
           "CONFIGURING":'running', 
           "FAILED":'failed',
           "TIMEOUT":'failed',
           "PREEMPTED":'failed',
           "NODE_FAIL":'failed',
           "REVOKED":'failed',
           "SPECIAL_EXIT":'failed',
	   "OUT_OF_ME+":'failed',
           "":'running'}


print(map_state[state])
