import os
import subprocess

# Populate AKAZE-based programs
programs = ['akaze_features', 'akaze_match', 'akaze_compare']
programs = [os.path.join('.', 'bin', prog) for prog in programs]

# Helper functions
def extract_AKAZE_features(imagePath):
    subprocess.run([programs[0], imagePath], check=True)

def match_AKAZE_features(imagePath1, imagePath2, gdTruthHomography):
    subprocess.run([programs[1], imagePath1, imagePath2, gdTruthHomography], check=True)

def compare_AKAZE_BRISK_ORB(imagePath1, imagePath2, gdTruthHomography):
    subprocess.run([programs[2], imagePath1, imagePath2, gdTruthHomography], check=True)

# Example datasets
imagePath1 = os.path.join('.', 'datasets', 'iguazu', 'img1.pgm')
imagePath2 = os.path.join('.', 'datasets', 'iguazu', 'img4.pgm')
gdTruthHomography = os.path.join('.', 'datasets', 'iguazu', 'H1to4p')

# Go!
extract_AKAZE_features(imagePath1)
match_AKAZE_features(imagePath1, imagePath2, gdTruthHomography)
compare_AKAZE_BRISK_ORB(imagePath1, imagePath2, gdTruthHomography)

