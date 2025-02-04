# -*- coding: utf-8 -*-
'''

'''

import numpy as np
import gzip
import os
import pickle
import phlame.helper_functions as helper

#%% combine_positions.py

class CombinePositions():

    def __init__(self,
                 path_to_diversity_files,
                 path_to_ref):
        
        self.diversity_files = helper.parse_file_list(path_to_diversity_files)
        self.path_to_ref = path_to_ref

        self.chr_starts, self.genome_length, self.scaf_names = helper.genomestats(self.path_to_ref)

    def main(self):
        
        variant_position_ls = []
        for file_ in self.diversity_files:

            with gzip.open(file_, 'rb') as f:
                div = pickle.load(f)
                variant_pos = div['variant_pos']
            variant_position_ls.append(variant_pos)

        all_positions = self.combine_positions(variant_position_ls)

        self.all_positions = all_positions

        return all_positions

    def combine_positions(self,
                          positions_files_ls):
        
        #in_outgroup: booleans giving whether or not each sample is in the outgroup
        # print("Processing outgroup booleans...")
        # in_outgroup=[]
        # with open(path_to_outgroup_boolean_file) as file:
        #     for line in file:
        #         in_outgroup.append(line)
        # #Bool of samples to include
        # include = [not i for i in in_outgroup]
        
        # #Get positions on reference genome
        # [self.chr_starts,self.genome_length,self.scaf_names] = ghf.genomestats(REFGENOMEDIRECTORY)
        
        #Find positions with at least 1 fixed mutation relative to reference genome
        print('\n\nFinding positions with at least 1 fixed mutation...\n')
        
        # positions_files_ls=[]
        # with open(path_to_positions_files) as file:
        #     for line in file:
        #         positions_files_ls.append(line)
                
        
        cp = self.generate_positions_snakemake(positions_files_ls)
        print(f"Found {len(cp)} positions where provided vcfs called a fixed variant in at least one in-group sample \n")

        #Todo: Add candidate positions manually
        #Combine different types of positions
        allp = cp
                
        return allp

    def generate_positions_single_sample(self,
                                         path_to_variant_vcf,
                                         maxFQ=-30):
        '''
        Python version of generate_positions_single_sample_snakemake.m

        Args:
            path_to_variant_vcf (str): Path to .variant.vcf.gz file.
            maxFQ (int): Purity threshold for including position.
            outgroup_bool (bool): Whether this sample is outgroup or not.

        Returns:
            None.

        '''    
        # print(f"Currently examining the following vcf file: {path_to_variant_vcf}\n")
        # print(f"FQ threshold: {int(maxFQ)}")
        
        # Initialize boolean vector for positions to include as candidate SNPs that
        # vary from the reference genome
        include = np.zeros((self.genome_length,1))
        
        
        f = gzip.open(path_to_variant_vcf,'rt')
        
        for line in f:
            if not line.startswith("#"):
                lineinfo = line.strip().split('\t')
                
                chromo=lineinfo[0]
                position_on_chr=lineinfo[1] #1-indexed
                
                if len(self.chr_starts) == 1:
                    position=int(lineinfo[1])
                else:
                    if chromo not in self.scaf_names:
                        raise ValueError("Scaffold name in vcf file not found in reference")
                    position=int(self.chr_starts[np.where(chromo==self.scaf_names)]) + int(position_on_chr)
                    #self.chr_starts begins at 0
                    
                alt=lineinfo[4]
                ref=lineinfo[3]
                
                #only consider for simple calls (not indel, not ambiguous)
                if (alt) and ("," not in alt) and (len(alt) == len(ref)) and (len(ref)==1):
                    #find and parse quality score
                    xt = lineinfo[7]
                    xtinfo = xt.split(';')
                    entrywithFQ=[x for x in xtinfo if x.startswith('FQ')][0]
                    fq=entrywithFQ[entrywithFQ.index("=")+1:]
                    
                    if float(fq) < maxFQ: #better than maxFQ
                        include[position-1]=1
                        #-1 converts position (1-indexed) to index
        
        #+1 converts index back to position for p2chrpos
        var_positions=helper.p2chrpos(np.nonzero(include)[0]+1,self.chr_starts)
        
        #save
        # with gzip.open(path_to_output_positions,"wb") as f:
        #     pickle.dump(Var_positions,f)
            
        # print(f"{len(Var_positions)} variable positions found passing quality threshold")
        
        return var_positions


    def chrpos2index(self, chrpos):
        '''Python version of chrpos2index.m

        Args:
            chrpos (arr): px2 array of position and chromsome idx.
            self.chr_starts (arr): Vector of chromosome starts (begins at 0).

        Returns:
            p (arr): Vector of position indexes.

        '''
        if np.size(chrpos,0) < np.size(chrpos,1):
            chrpos=chrpos.T
            print('Reversed orientation of chrpos')
            
        if len(self.chr_starts) == 1:
            p=chrpos[:,1]
        else:
            p=self.chr_starts[chrpos[:,0]-1]+chrpos[:,1]

        return p


    def generate_positions_snakemake(self, positions_list):
        '''Python version of generate_positions_snakemake.m
        
        Args:
            paths_to_input_p_files (list): List of input positions files.
            REFGENOMEDIRECTORY (str): Path to reference genome.

        Returns:
            combined_pos (arr): Vector of variable positions across samples.

        '''
                
        # initialize vector to count occurrances of variants across samples
        timesvariant = np.zeros((self.genome_length,1))
        
        # for i in range(len(positions_files_list)):
        #     #load in positions array for sample
        #     with gzip.open(positions_files_list[i].rstrip('\n'),"rb") as f:
        #     # with open(positions_files_list[i].rstrip('\n'),"rb") as f:
        #         positions=pickle.load(f)
            
        #     if len(positions)>2:
        #         x=self.chrpos2index(positions)
                
        #         timesvariant[x]=timesvariant[x]+1

        for positions in positions_list:

            if len(positions) > 2:
                x = self.chrpos2index(positions)
                timesvariant[x] = timesvariant[x] + 1
        
        
        #Keep positions that vary from the reference in at least one sample but
        #that don't vary from the reference in ALL samples
        combined_pos = np.where((timesvariant > 0) & (timesvariant < len(positions_list)))[0]
        
        return combined_pos

#%% pileup2diversity.py

class Pileup2Diversity:

    def __init__(self, 
                 path_to_pileup,
                 path_to_vcf,
                 path_to_variant_vcf,
                 path_to_ref,
                 path_to_output_diversity=None):
        
        self.path_to_pileup = path_to_pileup
        self.path_to_vcf = path_to_vcf
        self.path_to_variant_vcf = path_to_variant_vcf

        self.path_to_output_diversity = path_to_output_diversity

        self.chr_starts, self.genome_length, self.scaf_names = helper.genomestats(path_to_ref)

    def main(self):

        data, coverage = self.pileup2diversity(self.path_to_pileup,
                                               self.chr_starts,
                                               self.genome_length,
                                               self.scaf_names)
        

        quals_sample = Vcf2Quals.vcf_to_quals(self.path_to_vcf,
                                              self.chr_starts,
                                              self.genome_length,
                                              self.scaf_names)
        
        variant_pos = self.generate_positions_single_sample(self.path_to_variant_vcf)
        
        # print(variant_pos)

        if self.path_to_output_diversity:

            self.write_diversity(data, quals_sample, variant_pos,
                                 self.path_to_output_diversity)

    @staticmethod
    def write_diversity(data,
                        quals_sample,
                        variant_pos, path_to_output_diversity):
        
        diversity = {'data': data,
                     'quals': quals_sample,
                     'variant_pos': variant_pos}
        
        print(diversity['variant_pos'])

        with gzip.open(path_to_output_diversity, 'wb') as f:
            
            pickle.dump(diversity, f)

    @staticmethod
    def pileup2diversity(path_to_pileup,
                         chr_starts, genome_length, scaf_names):
        """Grabs relevant allele info from mpileupfile and stores as a nice array
        This version ONLY records counts and indel_counter

        Args:
            input_pileup (str): Path to input pileup file.
            path_to_ref (str): Path to reference genome file
            
        """
        #parameters
        nts = 'ATCGatcg'
        nts_dict = {'A':0,'T':1,'C':2,'G':3,'a':4,'t':5,'c':6,'g':7}
        num_fields=10
        indelregion=3 #region surrounding each p where indels recorded 
        #get reference genome + position information
        
        #initialize output array
        data = np.zeros((genome_length,num_fields)) #format [[A T C G  a t c g],[...]]
        
        #read in mpileup file
        print(f"Reading input file: {path_to_pileup}")
        mpileup = open(path_to_pileup)
        
        #####
        loading_bar=0
        
        for line in mpileup:
            
            # loading_bar+=1
            # if loading_bar % 50000 == 0:
            #     print('.')
            
            lineinfo = line.strip().split('\t')
            
            #holds info for each position before storing in data
            temp = np.zeros((num_fields))
            
            chromo = lineinfo[0]
            #position (absolute)
            if len(chr_starts) == 1:
                position=int(lineinfo[1])
            else:
                if chromo not in scaf_names:
                    raise ValueError("Scaffold name in pileup file not found in reference")
                position=int(chr_starts[np.where(chromo==scaf_names)]) + int(lineinfo[1])
                #chr_starts begins at 0
            
            #ref allele
            ref=nts_dict[lineinfo[2]] # convert to 0123
            if ref > 4:
                ref = ref - 4
            
            #calls info
            #calls=lineinfo[4]
            calls=np.fromstring(lineinfo[4], dtype=np.int8) #to ASCII
            # spits out a warning
            # calls=np.array([ord(l) for l in lineinfo[4]]) #ASCII
            
            #find starts of reads ('^' in mpileup)
            startsk=np.where(calls==94)[0]
            for k in startsk:
                calls[k:k+2]=-1 #WHAT IS -1
                #remove mapping character, absolutely required because the next chracter could be $
            
            #find ends of reads ('$' in mpileup)
            endsk=np.where(calls==36)[0]
            calls[endsk]=-1
            
            #find indels + calls from reads supporting indels ('+-')
            indelk = np.where((calls==43) | (calls==45))[0]
            for k in indelk:
                if (calls[k+2] >=48) and (calls[k+2] < 58): #2 digit indel (size > 9 and < 100)
                    indelsize=int(chr(calls[k+1]) + chr(calls[k+2])) 
                    #indelsize=str2double(char(calls(k+1:k+2))); MATLAB
                    indeld=2
                else: #1 digit indel (size <= 9)
                    indelsize=int(chr(calls[k+1]))
                    indeld=1
                #record that indel was found in +/- indelregion nearby
                #indexing is slightly different here from matlab version
                if calls[k]==45: #deletion
                    if (position-indelregion-1 >= 0) and (position+indelsize+indelregion-1 < genome_length):
                        #must store directly into data as it affects lines earlier and later
                        data[position-indelregion-1:position+indelsize+indelregion-1,9]+=1
                    elif position-indelregion >= 0: #for indels at end
                        data[position-indelregion-1:,9]+=1
                    else: #for indels at beg
                        data[:position+indelsize+indelregion-1,9]+=1
                else: #insertion
                    #insertion isn't indexed on the chromosome, no need for complex stuff
                    if (position-indelregion-1 >= 0) and (position+indelregion-1 < genome_length):
                        data[position-indelregion-1:position+indelregion-1,8]+=1
                    elif position-indelregion >= 0:
                        data[position-indelregion-1:,8]+=1
                    else:
                        data[:position+indelsize-1,8]+=1

                #remove indel info from counting
                calls[k:(k+1+indeld+indelsize)] = -1 #don't remove base that precedes an indel
            
            #replace reference matches (.,) with their actual calls
            if ref >=0:
                calls[np.where(calls==46)[0]]=ord(nts[ref]) #'.'
                calls[np.where(calls==44)[0]]=ord(nts[ref+4]) #','

            #index reads for finding scores
            simplecalls=calls[np.where(calls>0)[0]]
            #simplecalls is a tform of calls where each calls position
            #corresponds to its position in bq, mq, td
            
            #count how many of each nt and average scores

            for nt in range(8):
                nt_count=np.count_nonzero(simplecalls == ord(nts[nt]))
                if nt_count > 0:
                    temp[nt]=nt_count
            
            #-1 is needed to turn 1-indexed mpileup to 0-indexed arr
            data[position-1,0:8]=temp[0:8]
            
        #######
        mpileup.close()
        
        #calc coverage
        coverage=np.sum(data[:,0:8],1)
        
        return data, coverage
    
    def generate_positions_single_sample(self,
                                         path_to_variant_vcf,
                                         maxFQ=-30):
        '''
        Python version of generate_positions_single_sample_snakemake.m

        Args:
            path_to_variant_vcf (str): Path to .variant.vcf.gz file.
            maxFQ (int): Purity threshold for including position.
            outgroup_bool (bool): Whether this sample is outgroup or not.

        Returns:
            None.

        '''    
        # print(f"Currently examining the following vcf file: {path_to_variant_vcf}\n")
        # print(f"FQ threshold: {int(maxFQ)}")
        
        # Initialize boolean vector for positions to include as candidate SNPs that
        # vary from the reference genome
        include = np.zeros((self.genome_length,1))
        
        
        f = gzip.open(path_to_variant_vcf,'rt')
        
        for line in f:
            if not line.startswith("#"):
                lineinfo = line.strip().split('\t')
                
                chromo=lineinfo[0]
                position_on_chr=lineinfo[1] #1-indexed
                
                if len(self.chr_starts) == 1:
                    position=int(lineinfo[1])
                else:
                    if chromo not in self.scaf_names:
                        raise ValueError("Scaffold name in vcf file not found in reference")
                    position=int(self.chr_starts[np.where(chromo==self.scaf_names)]) + int(position_on_chr)
                    #self.chr_starts begins at 0
                    
                alt=lineinfo[4]
                ref=lineinfo[3]
                
                #only consider for simple calls (not indel, not ambiguous)
                if (alt) and ("," not in alt) and (len(alt) == len(ref)) and (len(ref)==1):
                    #find and parse quality score
                    xt = lineinfo[7]
                    xtinfo = xt.split(';')
                    entrywithFQ=[x for x in xtinfo if x.startswith('FQ')][0]
                    fq=entrywithFQ[entrywithFQ.index("=")+1:]
                    
                    if float(fq) < maxFQ: #better than maxFQ
                        include[position-1]=1
                        #-1 converts position (1-indexed) to index
        
        #+1 converts index back to position for p2chrpos
        var_positions=helper.p2chrpos(np.nonzero(include)[0]+1,self.chr_starts)
        
        #save
        # with gzip.open(path_to_output_positions,"wb") as f:
        #     pickle.dump(Var_positions,f)
            
        # print(f"{len(Var_positions)} variable positions found passing quality threshold")
        
        return var_positions


class Vcf2Quals:

    @staticmethod
    def vcf_to_quals(path_to_vcf_file,
                     chr_starts, genome_length, scaf_names):
        '''
        Python version of vcf_to_quals_snakemake.py
        Given a vcf file with one file per line, grabs FQ score for each positions. Ignores lines corresponding to indels

        Args:
            path_to_vcf_file (str): Path to .vcf file.
            output_path_to_quals (str): Path to output quals file
            REFGENOMEDIRECTORY (str): Path to reference genome directory.

        Returns:
            None.
            
        '''

        #initialize vector to record quals
        quals = np.zeros((genome_length,1), dtype=int)
        
        print(f"Loaded: {path_to_vcf_file}")
        file = gzip.open(path_to_vcf_file,'rt') #load in file
        
        for line in file:
            if not line.startswith("#"):
                lineinfo = line.strip().split('\t')
                
                #Note: not coding the loading bar in the matlab script
                
                chromo=lineinfo[0]
                position_on_chr=lineinfo[1] #1-indexed
                
                if len(chr_starts) == 1:
                    position=int(lineinfo[1])
                else:
                    if chromo not in scaf_names:
                        raise ValueError("Scaffold name in vcf file not found in reference")
                    position=int(chr_starts[np.where(chromo==scaf_names)]) + int(position_on_chr)
                    #chr_starts begins at 0
                    
                alt=lineinfo[4]
                ref=lineinfo[3]
                
                #only consider for simple calls (not indel, not ambiguous)
                if (alt) and ("," not in alt) and (len(alt) == len(ref)) and (len(ref)==1):
                    #find and parse quality score
                    xt = lineinfo[7]
                    xtinfo = xt.split(';')
                    entrywithFQ=[x for x in xtinfo if x.startswith('FQ')][0]
                    fq=float(entrywithFQ[entrywithFQ.index("=")+1:])
                    
                    #If already a position wiht a stronger FQ here, don;t include this
                    #More negative is stronger
                    if fq < quals[position-1]:
                        quals[position-1]=round(fq) 
                            #python int(fq) will by default round down, round matches matlab behavior
                            #-1 important to convert position (1-indexed) to python index
        
        return quals

#%%

class Case():

    def __init__(self,
                 path_to_sample_names,
                 path_to_diversity_files,
                 path_to_ref,
                 path_to_out_cmt):
        
        self.path_to_sample_names_file = path_to_sample_names
        
        self.path_to_diversity_files = path_to_diversity_files
        self.diversity_files = helper.parse_file_list(self.path_to_diversity_files)

        self.path_to_out_cmt = path_to_out_cmt
        self.path_to_ref = path_to_ref

        # self.sample_names_check()

        self.chr_starts, self.genome_length, self.scaf_names = helper.genomestats(self.path_to_ref)

    # def sample_names_check(self):

    #     counter = 0
    #     for i, sample_name in enumerate(self.sample_names):

    #         while counter <= 3:
    #             if not (sample_name in str(self.diversity_files[i])):
    #                 print(f"Warning! Sample name {sample_name} not found in the name of corresponding diversity file {self.diversity_files[i]}. Continuing...")
    #                 counter += 1

    #         if counter > 3:
    #             print("Not printing any further warnings...")
            
    def main(self):

        # Get sample names
        print('Processing sample names...')
        with open(self.path_to_sample_names_file, 'r') as f:
            self.sample_names = f.read().splitlines()
        nsamples = len(self.sample_names)  # save number of samples
        print('Total number of samples: ' + str(nsamples))


        # Get all candidate SNP positions
        print('Processing candidate SNP positions...')
        combine = CombinePositions(self.path_to_diversity_files, self.path_to_ref)
        self.p = combine.main()


        dim=8

        self.counts = np.zeros((dim, len(self.p), nsamples), dtype='uint')  # initialize
        self.indel_counter = np.zeros((2, len(self.p), nsamples), dtype='uint')
        self.quals = np.zeros((len(self.p), nsamples), dtype='int')

        print('Processing diversity data...')
        for i, diversity_file in enumerate(self.diversity_files):

            with gzip.open(diversity_file, 'rb') as f:
                diversity = pickle.load(f)

                data = diversity['data']
                quals_sample = diversity['quals'].flatten()

            self.counts[:, :, i] = data[self.p - 1, 0:dim].T  # -1 convert position to index
            self.indel_counter[:, :, i] = data[self.p - 1, 8:10].T  # Num reads supporting indels and reads supporting deletions
            self.quals[:, i] = quals_sample[self.p - 1]  # -1 is to convert position to index

        # ## counts: counts for each base from forward and reverse reads at each candidate position for all samples
        # print('Gathering counts data at each candidate position...\n')

        # # Import list of directories for where to diversity file for each sample
        # dim=8
        # self.counts = np.zeros((dim, len(self.p), nsamples), dtype='uint')  # initialize
        # self.indel_counter = np.zeros((2, len(self.p), nsamples), dtype='uint')

        # for i, mpileup_file in enumerate(self.path_to_mpileup_files):

        #     data, coverage = Pileup2Diversity.pileup2diversity(mpileup_file,
        #                                                        self.chr_starts,
        #                                                        self.genome_length,
        #                                                        self.scaf_names)

        #     self.counts[:, :, i] = data[self.p - 1, 0:dim].T  # -1 convert position to index
        #     self.indel_counter[:, :, i] = data[self.p - 1, 8:10].T  # Num reads supporting indels and reads supporting deletions


        # ## Quals: quality score (relating to sample purity) at each position for all samples
        # print('Gathering quality scores at each candidate position...')
        # # Import list of directories for where to quals for each sample

        # self.quals = np.zeros((len(self.p), nsamples), dtype='int')  # initialize

        # for i, vcf_file in enumerate(self.path_to_vcf_files):
            
        #     quals_sample = Vcf2Quals.vcf_to_quals(vcf_file,
        #                                           self.chr_starts,
        #                                           self.genome_length,
        #                                           self.scaf_names)
                    

        #     quals_sample = quals_sample.flatten()
        #     self.quals[:, i] = quals_sample[self.p - 1]  # -1 is to convert position to index


        self.write_CMT()

    def write_CMT(self):

        outdir = os.path.dirname(self.path_to_out_cmt)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        CMT = {'sample_names': self.sample_names,
               'p': self.p,
               'counts': self.counts,
               'quals': self.quals,
               'indel_counter': self.indel_counter}

        file_path = self.path_to_out_cmt
        print("Saving " + self.path_to_out_cmt)
        with gzip.open(file_path, 'wb') as f:
            pickle.dump(CMT, f)

    # @staticmethod
    # def parse_file_list(path_to_file):
    #     with open(path_to_file, 'r') as f:
    #         file_list = f.read().splitlines()

    #     for file_ in file_list:
    #         if not os.path.isabs(file_):
    #             abs_path = os.path.abspath(file_)
    #             if not os.path.exists(abs_path):
    #                 raise FileNotFoundError(f"File not found: {abs_path}")
    #             file_list[file_list.index(file_)] = os.path.abspath(file_)

    #     return file_list
