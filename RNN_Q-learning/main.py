import tensorflow as tf
import numpy as np
import sys
from magenta.models.rl_tuner import rl_tuner
from magenta.models.rl_tuner import rl_tuner_ops

# Model parameter settings
SAVE_PATH = "/tmp/rl_tuner/"
ALGORITHM = 'q'
REWARD_SCALER = 1
OUTPUT_EVERY_NTH = 50000
NUM_NOTES_IN_COMPOSITION = 32
PRIME_WITH_MIDI = True
NOTE_RNN_CHECKPOINT_FILE = 'note_rnn.ckpt'
MIDI_PRIMER = 'bach_846.mid'
PRIMING_MODE = 'single_midi'
NOTE_RNN_TYPE = 'default'

# Creates RLTuner object with above specified parameters. 
rl_net = rl_tuner.RLTuner(SAVE_PATH,
                          # Hyperparameters
                          reward_scaler=REWARD_SCALER,
                          priming_mode=PRIMING_MODE,
                          algorithm=ALGORITHM,
                          note_rnn_checkpoint_file=NOTE_RNN_CHECKPOINT_FILE,
                          note_rnn_type=NOTE_RNN_TYPE,
                          # Other music related settings.
                          num_notes_in_melody=NUM_NOTES_IN_COMPOSITION,
                          input_size=rl_tuner_ops.NUM_CLASSES,
                          num_actions=rl_tuner_ops.NUM_CLASSES,
                          midi_primer=MIDI_PRIMER,
                          # Logistics.
                          output_every_nth=OUTPUT_EVERY_NTH)

# Generate and display music sequence
rl_net.generate_music_sequence(visualize_probs=True, title='post_rl')