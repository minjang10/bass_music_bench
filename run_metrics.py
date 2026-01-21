from datetime import datetime
import subprocess
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structural-segmentation', type=str, required=True)
    parser.add_argument('--musicological-analysis', type=str, required=True)
    parser.add_argument('--artist-collab', type=str, required=True)
    # parser.add_argument('--lyrics-transcription', type=str, required=True)
    args = parser.parse_args()
    structural_segmentation_result_file = args.structural_segmentation
    musicological_analysis_result_file = args.musicological_analysis
    artist_collab_result_file = args.artist_collab
    # lyrics_transcription_result_file = args.lyrics_transcription

    # Run the evaluation scripts
    subprocess.run(['python', 'collab_analysis_evaluation.py', '--result-file', artist_collab_result_file])
    print('=' * 100)
    subprocess.run(['python', 'structural_segmentation_evaluation.py', '--result-file', structural_segmentation_result_file])
    print('=' * 100)
    subprocess.run(['python', 'musicological_analysis_evaluation.py', '--result-file', musicological_analysis_result_file])
    # print('=' * 100)
    # subprocess.run(['python', 'lyrics_transcription_evaluation.py', '--result-file', lyrics_transcription_result_file])

if __name__ == '__main__':
    main()