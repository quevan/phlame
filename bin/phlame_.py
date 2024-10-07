#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:57:25 2022

@author: evanqu
"""
import argparse
import os
import sys

# sys.set_int_max_str_digits(0)
# So I don't get weird ValueError: Exceeds the limit (4300) for integer string conversion

import phlame.classify_module as classify
import phlame.tree_module as tree
import phlame.helper_functions as helper
import phlame.make_classifier_module as makedb

#%%

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('phlame.py'), '../phlame')))


def print_help():
    print('')
    print('            ...::: PhLAMe v1.0 :::...''')
    print('       Evan Qu, Lieberman Lab, MIT. 2023\n''')
    print('Usage: phlame.py [-h] {classify,makedb,tree} ...\n')
    print('''
Choose one of the operations below for more detailed help.
Example: phlame classify -h

Main operations:
    classify ->  Determine lineage-level frequencies within a metagenome using a PhLAMe database
    makedb   ->  Create a PhLAMe database
    tree     ->  Identify candidate lineages from a phylogenetic tree

Auxiliary operations:
    TBD
            ''')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog = 'phlame',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    
    subparsers = parser.add_subparsers(help='Desired operation',dest='operation')
    
    classify_op = subparsers.add_parser('classify')
    makedb_op = subparsers.add_parser('makedb')
    tree_op = subparsers.add_parser('tree')

    # Classify arguments
    classify_op.add_argument('-i', dest='input', type=str, required=True, 
                          help='Path to input counts file')
    classify_op.add_argument('-c', dest='classifier', type=str, required=True,
                          help='Path to classifer file')
    classify_op.add_argument('-l', dest='level', type=str, default=False, required=False,
                          help='Level specification')
    classify_op.add_argument('-o', dest='output', type=str, required=True,
                          help='Path to output frequencies file (.csv)')
    classify_op.add_argument('-p', dest='outputdata', type=str, default=False, required=False,
                          help='Path to output data file (.pickle)')
    classify_op.add_argument('--max_pi', type=float, default=0.3, required=False,
                          help='Maximum pi value to count a lineage as present')
    classify_op.add_argument('--min_prob', type=float, default=0.5, required=False,
                          help='Minimum probability score to count a lineage as present')
    classify_op.add_argument('--min_snps', type=int, default=10, required=False,
                          help='Minimum number of marker SNPs with non-zero counts to count a lineage as present')
    classify_op.add_argument('--min_hpd', type=int, default=10, required=False,
                          help='Minimum value the highest posterior density interval over divergence must cover to count a lineage as present')
    classify_op.add_argument('--seed', type=int, required=False, default=False,
                          help='Set random seed for reproducibility.') 


    # Make_classifier arguments
    makedb_op.add_argument('-i', dest='input', type=str, required=True, 
                          help='Path to input candidate mutation table')
    makedb_op.add_argument('-c', dest='clades', type=str, required=True,
                          help='Path to candidate clades file')
    makedb_op.add_argument('-o', dest='output', type=str, required=True,
                          help='Path to output classifier')
    makedb_op.add_argument('-r', dest='cladestree', default=False, type=bool, required=False,
                          help='Path to candidate clades tree file (newick format)')
    makedb_op.add_argument('--min_snps', type=int, default=10, required=False,
                          help='Minimum number of SNPs to include a candidate clade')
    makedb_op.add_argument('--maxn', type=float, default=0.1, required=False,
                          help='Maximum percentage of Ns for a position to be considered')
    makedb_op.add_argument('--core', type=float, default=0.9, required=False,
                          help='Maximum number of non-Ns across isolates for a position to be considered')
    makedb_op.add_argument('--outgroup', type=float, default=False, required=False,
                          help='Maximum number of non-Ns across isolates for a position to be considered')

    # Tree arguments
    tree_op.add_argument('-i', dest='intree', type=str, required=True, 
                          help='Path to input phylogeny (Newick format)')
    tree_op.add_argument('-c', dest='incmt', type=str, default=False, required=False, 
                          help='Path to input candidate mutation table')
    tree_op.add_argument('-o', dest='outtree', type=str, required=True, 
                          help='Path to output called phylogeny (Newick format)')
    tree_op.add_argument('-p', dest='outclades', type=str, default=False, required=False, 
                          help='Path to output clade IDs file')

    tree_op.add_argument('--min_branch', type=float, default=100, required=False,
                          help='Minimum branch length leading up to a clade')
    tree_op.add_argument('--min_leaves', type=int, default=3, required=False,
                          help='Minimum number of leaves in a clade')
    tree_op.add_argument('--min_support', type=float, default=0.75, required=False,
                          help='Minimum bootstrap support for a clade')
    tree_op.add_argument('--rescale', required=False,  action='store_true',
                          help='Scale tree branch lengths to number of core-genome substitutions.')

    args = parser.parse_args()
    
    # Main help message
    if (len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help'):
            print_help()
            sys.exit(0)
    
    if args.operation=='classify':
        
        results = classify.Classify(args.input,
                                    args.classifier,
                                    args.output,
                                    level_input = args.level,
                                    path_to_output_data = args.outputdata,
                                    max_pi = args.max_pi,
                                    min_snps = args.min_snps,
                                    min_prob = args.min_prob,
                                    min_hpd=args.min_hpd,
                                    seed = args.seed)
        
        results.main()

    if args.operation=='makedb':

        db = makedb.MakeDB(args.input,
                           args.clades,
                           args.output,
                           path_to_candidate_clades_tree = args.cladestree,
                           min_cssnps = args.min_snps,
                           maxn = args.maxn,
                           core = args.core,
                           max_outgroup = args.outgroup,
                           min_maf_for_call = 0.75,
                           min_strand_cov_for_call = 2,
                           max_qual_for_call = -30)

        db.main()

        # makedb.make_classifier(args.input, args.output,
        #                         args.clades, args.cladestree,
        #                         args.min_snps,
        #                         args.maxn,
        #                         args.core)
        
    if args.operation=='tree':
        
        results = tree.Tree(args.intree,
                            args.outtree,
                            args.outclades,
                            min_branch_len=args.min_branch,
                            min_nsamples=args.min_leaves,
                            min_support=args.min_support,
                            path_to_cmt_file=args.incmt,
                            rescale=args.rescale)
        
        results.main()

        
        # print("Pass to classify operation with params:")
        # print(f"path_to_cts_file={args.input}")
        # print(f"path_to_classifier={args.classifier}")
        # print(f"level_input={args.level}")
        # print(f"path_to_output_frequencies={args.output}")
        # print(f"path_to_output_data={args.outputdata}")
        # print(f"max_pi={args.max_pi}")
        # print(f"min_prob={args.min_prob}")
        # print(f"min_snps={args.min_snps}")