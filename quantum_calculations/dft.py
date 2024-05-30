import os
import subprocess
import time
import hashlib
import numpy as np
import theodore
import csv

from UncAGA.quantum_calculations.xtb import XTB_Processor
from UncAGA.quantum_calculations.postprocessing import theodore_analysis

class DFTBatchEvaluator:  
    def __init__(self, list_chromosomes, output_suffix,config):
        self.list_chromosomes = list(set(list_chromosomes))
        self.output_suffix = output_suffix
        self.config = config
        self.results_path = os.path.join(self.config.output_path, 'DFT_calculations_active')
        os.makedirs(self.results_path, exist_ok=True)
        
        self.list_chrom_hashes=[]
        self._process_chromosomes()

        
    def _process_chromosomes(self):
        for chromosome in self.list_chromosomes:
            xtb_processor = XTB_Processor(chromosome,self.config,"DFT_calculations_active")
            print(chromosome)
            chrom_hash = xtb_processor.generate_smiles_hash()
            self.list_chrom_hashes.append(chrom_hash)
            xtb_processor.create_xyz_with_xtb()

        self.list_chromosomes, self.list_chrom_hashes = self._filter_successful_xtb()
        
        # TODO remove
        assert len(self.list_chromosomes) == len(self.list_chrom_hashes)
        
        self._launch_gaussian_GS_relaxation()
        self.relaxation_finished = self._check_process_completion('GS_relaxation')
        self._convert_log_to_xyz()
        self.conversion_finished = self._check_process_completion('conversion')
        self._run_tddft()
        self.tddft_finished = self._check_process_completion('tddft')
        results = self.read_tddft_log_files_write_csv()

    def _filter_successful_xtb(self):
        # Implementation of filtering successful xtb results
        # This is a placeholder implementation. Replace it with actual logic.
        return self.list_chromosomes, self.list_chrom_hashes

    def _launch_gaussian_GS_relaxation(self):
        # Placeholder for launching Gaussian relaxation
        for chrom_hash in self.list_chrom_hashes:
            xyz_file = os.path.join(self.results_path, f"{chrom_hash}_pysis_conv.xyz")
            subprocess.run(['sh', self.config.run_gaussian_relaxation, xyz_file])


    def _convert_log_to_xyz(self):
        # Placeholder for converting log files to XYZ format
        for chrom_hash in self.list_chrom_hashes:
            log_file = os.path.join(self.results_path, f"{chrom_hash}_pysis_conv.log")
            try:
                self.write_xyz_from_gaussian_log(log_file)
            except FileNotFoundError:
                self.list_chromosomes.remove(chrom_hash)
                self.list_chrom_hashes.remove(chrom_hash)

    def _run_tddft(self):
        # Placeholder for running TDDFT calculations
        for chrom_hash in self.list_chrom_hashes:
            xyz_file_relaxed = os.path.join(self.results_path, f"{chrom_hash}_pysis_conv_relaxed.xyz")
            subprocess.run(['sh', self.config.run_gaussian_tddft, xyz_file_relaxed])
            
    def _check_process_completion(self,process_type):
        """
        Check if a specified process (Gaussian relaxation or XYZ conversion) is completed.

        Parameters:
        - process_type (str): Type of process to check ('GS_relaxation' or 'conversion' or 'tddft').

        Returns:
        - (list): A list of booleans indicating completion status for each chromosome.
        """

        if process_type == 'GS_relaxation':
            max_hours = self.config.max_nr_hours_relaxation
            file_suffix = '_pysis_conv.log'
            completion_indicator = 'Elapsed time'
        elif process_type == 'conversion':
            max_hours = self.config.max_nr_hours_conversion
            file_suffix = '_pysis_conv_relaxed.xyz'
            completion_indicator = None
        elif process_type == 'tddft':
            max_hours = self.config.max_nr_hours_tddft
            file_suffix = '_pysis_conv_relaxed.log'
            completion_indicator = None
        else:
            raise ValueError("Invalid process type specified.")

        process_finished = [False] * len(self.list_chrom_hashes)
        t_end = time.time() + 60 * 60 * max_hours

        while time.time() < t_end:
            for idx, chrom_hash in enumerate(self.list_chrom_hashes):
                file_path = os.path.join(self.results_path, f"{chrom_hash}{file_suffix}")
                if os.path.exists(file_path):
                    if completion_indicator:
                        with open(file_path, 'r') as f:
                            for line in f:
                                if completion_indicator in line:
                                    process_finished[idx] = True
                    else:
                        process_finished[idx] = True

            if all(process_finished):
                break

            time.sleep(10)

        return process_finished

    def read_tddft_log_files_write_csv(self):
        for idx, chrom_hash in enumerate(self.list_chrom_hashes):
            log_file_relaxed = os.path.join(self.results_path, f"{chrom_hash}_pysis_conv_relaxed.log")
            self.process_log_file(log_file_relaxed,idx)
            
            

    def process_log_file(self, log_file_relaxed,idx):
        T1, S1, T2, osc_strength_S1, S1ehdist = (np.nan,) * 5

        print("READ OUT RESULTS", self.tddft_finished,log_file_relaxed,idx)
        if self.tddft_finished:  
            normal_termin, singlet_energies, triplet_energies, osc_strengths_singlets = False, [], [], []

            with open(log_file_relaxed) as f:
                for line in f:
                    if 'Singlet-' in line:
                        singlet_energies.append(float(line.split()[4]))
                        osc_strengths_singlets.append(line.split()[8])

                    if 'Triplet-' in line:
                        triplet_energies.append(float(line.split()[4]))

                    if 'Normal termination' in line:
                        print("FOUND Normal termination")
                        normal_termin = True
            print('Singlet', singlet_energies)
            print('normal_termin', normal_termin)

            if normal_termin:
                try:
                    print('triplet', triplet_energies)
                    print('singlet', singlet_energies)
                    S1 = singlet_energies[0]
                    T1 = triplet_energies[0]
                    T2 = triplet_energies[1]
                    osc_strength_S1 = float(osc_strengths_singlets[0].split("=")[1])
                except IndexError:
                    print("INDEX ERROR")
        
            try:
            #TODO CHANGE
                S1ehdist = theodore_analysis.theodore_workflow_S1_excdist(self.config.path_dens_ana_in,log_file_relaxed)
            except theodore.error_handler.MsgError:
                print("Theodore error", flush=True)
                S1ehdist = None  

            results_file_path = os.path.join(self.config.output_path, f'DFT_results_{self.output_suffix}')
            with open(results_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.list_chromosomes[idx], self.list_chrom_hashes[idx], T1, S1, T2, S1ehdist, osc_strength_S1])
                #TODOOOO

    def write_xyz_from_gaussian_log(self,log_file):

        code = {"1" : "H", "2" : "He", "3" : "Li", "4" : "Be", "5" : "B", \
    "6"  : "C", "7"  : "N", "8"  : "O",  "9" : "F", "10" : "Ne", \
    "11" : "Na" , "12" : "Mg" , "13" : "Al" , "14" : "Si" , "15" : "P", \
    "16" : "S"  , "17" : "Cl" , "18" : "Ar" , "19" : "K"  , "20" : "Ca", \
    "21" : "Sc" , "22" : "Ti" , "23" : "V"  , "24" : "Cr" , "25" : "Mn", \
    "26" : "Fe" , "27" : "Co" , "28" : "Ni" , "29" : "Cu" , "30" : "Zn", \
    "31" : "Ga" , "32" : "Ge" , "33" : "As" , "34" : "Se" , "35" : "Br", \
    "36" : "Kr" , "37" : "Rb" , "38" : "Sr" , "39" : "Y"  , "40" : "Zr", \
    "41" : "Nb" , "42" : "Mo" , "43" : "Tc" , "44" : "Ru" , "45" : "Rh", \
    "46" : "Pd" , "47" : "Ag" , "48" : "Cd" , "49" : "In" , "50" : "Sn", \
    "51" : "Sb" , "52" : "Te" , "53" : "I"  , "54" : "Xe" , "55" : "Cs", \
    "56" : "Ba" , "57" : "La" , "58" : "Ce" , "59" : "Pr" , "60" : "Nd", \
    "61" : "Pm" , "62" : "Sm" , "63" : "Eu" , "64" : "Gd" , "65" : "Tb", \
    "66" : "Dy" , "67" : "Ho" , "68" : "Er" , "69" : "Tm" , "70" : "Yb", \
    "71" : "Lu" , "72" : "Hf" , "73" : "Ta" , "74" : "W"  , "75" : "Re", \
    "76" : "Os" , "77" : "Ir" , "78" : "Pt" , "79" : "Au" , "80" : "Hg", \
    "81" : "Tl" , "82" : "Pb" , "83" : "Bi" , "84" : "Po" , "85" : "At", \
    "86" : "Rn" , "87" : "Fr" , "88" : "Ra" , "89" : "Ac" , "90" : "Th", \
    "91" : "Pa" , "92" : "U"  , "93" : "Np" , "94" : "Pu" , "95" : "Am", \
    "96" : "Cm" , "97" : "Bk" , "98" : "Cf" , "99" : "Es" ,"100" : "Fm", \
    "101": "Md" ,"102" : "No" ,"103" : "Lr" ,"104" : "Rf" ,"105" : "Db", \
    "106": "Sg" ,"107" : "Bh" ,"108" : "Hs" ,"109" : "Mt" ,"110" : "Ds", \
    "111": "Rg" ,"112" : "Uub","113" : "Uut","114" : "Uuq","115" : "Uup", \
    "116": "Uuh","117" : "Uus","118" : "Uuo"}




        # Read the Gaussian log file
        with open(log_file, 'r') as f:
            lines = f.readlines()

        # Find the last occurrence of the Standard orientation section
        start_line = None
        end_line = None
        for i in range(len(lines) - 1, -1, -1):
            if "Standard orientation" in lines[i]:
                start_line = i + 5
                break
        print(start_line)
        for j in range(start_line,len(lines)):
            if "------" in lines[j]:
                end_line = j
                break
        print(end_line)
        if start_line is None:
            print("Failed to find optimized molecular geometry in the log file.")
            return

        # Extract the Cartesian coordinates and element types
        cartesian_lines = lines[start_line:end_line]
        #print(cartesian_lines)
        xyz_coords = []
        for line in cartesian_lines:
            tokens = line.split()
            print(tokens)
            if len(tokens) == 6:
                atom_symbol = tokens[1]
                x, y, z = map(float, tokens[3:6])
                print("x,y,z",x,y,z,flush=True)
                xyz_coords.append(f"{code[atom_symbol]} {x:.6f} {y:.6f} {z:.6f}\n")

        # Save the XYZ coordinates to a new file
        output_file = log_file.replace('.log', '_relaxed.xyz')
        with open(output_file, 'w') as f:
            f.write(f"{len(xyz_coords)}\n")
            f.write("XYZ coordinates extracted from Gaussian relaxation log file\n")
            f.writelines(xyz_coords)

        print(f"XYZ coordinates extracted and saved to {output_file}")
#         
        
   




    

        


