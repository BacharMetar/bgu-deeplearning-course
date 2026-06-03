import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset


class LyricsMelodyDataset(Dataset):
    """
    A PyTorch Dataset that pairs word sequences from song lyrics
    with a fixed-size vector summarising the song's MIDI melody.

    Each sample is one sliding window of `sequence_length` words
    together with the next word as the prediction target.
    """

    def __init__(self, df, midi_dir, word2vec_model, sequence_length=10, existing_vocab=None):
        """
        Args:
            df              : pandas DataFrame with columns ['artist', 'song', 'lyrics'].
                              Lyrics lines are joined by ' & '.
            midi_dir        : path to the directory containing .mid files.
            word2vec_model  : a loaded gensim KeyedVectors object (300-dim).
            sequence_length : number of words fed as input at each step.
            existing_vocab  : optional vocab list from the training dataset.
                              Pass train_dataset.vocab when building the test
                              dataset so both share identical word-to-index
                              mappings. Words unseen in training map to <UNK>.
        """
        self.midi_dir        = midi_dir
        self.wv              = word2vec_model
        self.sequence_length = sequence_length

        # Cache melody vectors so each MIDI file is read from disk only once
        self.midi_cache = {}

        # Build a lookup table: normalised "artist - song" → filename.mid
        midi_files     = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        self.midi_map  = self._build_midi_map(midi_files)

        # ------------------------------------------------------------------ #
        # Pre-process every song: tokenise lyrics and locate its MIDI file.  #
        # song_data[i] = (word_list, midi_filename_or_None)                  #
        # ------------------------------------------------------------------ #
        self.song_data = []
        all_words      = []  # collect every word for vocabulary construction

        for _, row in df.iterrows():
            words         = self._tokenize(row['lyrics'])
            midi_filename = self._find_midi(row['artist'], row['song'])
            self.song_data.append((words, midi_filename))
            all_words.extend(words)

        # ------------------------------------------------------------------ #
        # Vocabulary: reuse the training vocab when given, build one if not. #
        #                                                                     #
        # Reusing the training vocab on the test set ensures every word maps  #
        # to the same index the model was trained on. Test words unseen       #
        # during training fall back to index 0 (<UNK>).                      #
        # ------------------------------------------------------------------ #
        if existing_vocab is not None:
            # Test-set path: derive word2idx directly from the provided list
            self.vocab    = existing_vocab
            self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        else:
            # Training path: build a fresh vocabulary from this dataset
            self.vocab, self.word2idx = self._build_vocab(all_words)

        # ------------------------------------------------------------------ #
        # Build the flat sample index.                                        #
        # samples[i] = (song_idx, word_start_pos)                            #
        # A valid sample needs sequence_length words + 1 target word.        #
        # ------------------------------------------------------------------ #
        self.samples = []
        for song_idx, (words, _) in enumerate(self.song_data):
            # Slide a window of (sequence_length + 1) words across each song
            for start in range(len(words) - sequence_length):
                self.samples.append((song_idx, start))

    # ---------------------------------------------------------------------- #
    #  Internal helpers                                                        #
    # ---------------------------------------------------------------------- #

    def _tokenize(self, lyrics):
        """
        Split lyrics on '&' (line separator), lowercase each line,
        strip punctuation, and return a flat list of word strings.

        The '&' is replaced by a special <EOL> token instead of being
        discarded. This lets the model learn when a line should end,
        which is the mechanism for enforcing line-length structure
        through the loss function (the model is penalised whenever it
        fails to predict <EOL> at the right position).

        Example:
            "hello world & goodbye moon &"
            → ['hello', 'world', '<EOL>', 'goodbye', 'moon', '<EOL>']
        """
        lines = [line.strip() for line in lyrics.split('&') if line.strip()]
        words = []
        for line in lines:
            # Keep only letters, digits, apostrophes and whitespace
            cleaned = re.sub(r"[^a-z0-9\s']", '', line.lower())
            words.extend(cleaned.split())
            # Mark the end of this line so the model learns line boundaries
            words.append('<EOL>')
        # Mark the end of the entire song so the model learns when to stop generating
        words.append('<EOS>')
        return words

    def _build_vocab(self, all_words):
        """
        Build a word → index mapping from every word seen in the dataset.
        Index 0 is reserved for <UNK> (unknown / OOV words).
        """
        counts  = Counter(all_words)
        # Sort alphabetically for reproducibility
        vocab   = ['<UNK>'] + sorted(counts.keys())
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        return vocab, word2idx

    def _build_midi_map(self, midi_files):
        """
        Build a dict mapping normalised song keys to .mid filenames.

        MIDI files are named 'Artist_Name_-_Song_Title.mid'.
        We normalise by replacing underscores with spaces and lowercasing,
        so 'Elton_John_-_Candle_In_The_Wind.mid' becomes
        'elton john - candle in the wind'.
        """
        mapping = {}
        for fname in midi_files:
            # Strip extension, replace underscores, lowercase
            key = fname[:-4].replace('_', ' ').lower()
            mapping[key] = fname
        return mapping

    def _find_midi(self, artist, song):
        """
        Look up the MIDI filename for a given artist and song title.
        Returns None if no matching file is found.

        Strategy:
          1. Exact match on 'artist - song' (both normalised to lowercase).
          2. Partial match: song title appears anywhere in the MIDI key.
        """
        exact_key = f"{artist.strip().lower()} - {song.strip().lower()}"
        if exact_key in self.midi_map:
            return self.midi_map[exact_key]

        # Fallback: look for the song title as a substring
        song_lower = song.strip().lower()
        for midi_key, fname in self.midi_map.items():
            if song_lower in midi_key:
                return fname

        return None  # no MIDI found for this song

    def _extract_melody_vector(self, midi_filename):
        """
        Load a MIDI file and summarise its main melody as a 128-dim vector.

        Steps:
          1. Pick the non-drum instrument with the most notes (main melody).
          2. Compute the piano roll: shape (128, time_steps),
             where 128 = one row per MIDI pitch and time_steps depends on
             the sampling rate (fs=10 frames/sec used here for speed).
          3. Average across the time axis → shape (128,).
             Each element represents the average activity of one pitch
             over the entire song, capturing which notes dominate.

        Returns a float32 tensor of shape (128,).
        Falls back to a zero vector if no MIDI file is found or loading fails.
        """
        # Return a zero vector when no MIDI file is associated with the song
        if midi_filename is None:
            return torch.zeros(128, dtype=torch.float32)

        # Return cached result to avoid redundant disk reads
        if midi_filename in self.midi_cache:
            return self.midi_cache[midi_filename]

        midi_path = os.path.join(self.midi_dir, midi_filename)
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)

            # Find the main melody: non-drum track with the most notes
            melody_track = None
            max_notes    = 0
            for instrument in midi.instruments:
                if not instrument.is_drum and len(instrument.notes) > max_notes:
                    max_notes    = len(instrument.notes)
                    melody_track = instrument

            if melody_track is None:
                # No melodic track found; return zeros
                vector = torch.zeros(128, dtype=torch.float32)
            else:
                # piano_roll shape: (128, time_steps)
                # fs=10 → 10 frames per second (low res is fine for a summary)
                piano_roll = melody_track.get_piano_roll(fs=10)

                # Mean over time axis → shape (128,)
                # Result: average velocity/activity per pitch across the song
                melody_np = piano_roll.mean(axis=1).astype(np.float32)
                vector    = torch.tensor(melody_np, dtype=torch.float32)

        except Exception as e:
            print(f"Warning: could not load MIDI '{midi_filename}': {e}")
            vector = torch.zeros(128, dtype=torch.float32)

        # Cache for future __getitem__ calls
        self.midi_cache[midi_filename] = vector
        return vector

    # ---------------------------------------------------------------------- #
    #  Dataset interface                                                       #
    # ---------------------------------------------------------------------- #

    def __len__(self):
        """Total number of (sequence, target) samples across all songs."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one training sample as a dict:

          'text_sequence' : float32 tensor, shape (sequence_length, 300)
                            Word2Vec embedding for each word in the window.
                            Zero vector for OOV words.
§
          'target_word'   : int — vocabulary index of the word that follows
                            the input window (what the model must predict).

          'melody_vector' : float32 tensor, shape (128,)
                            Mean piano-roll activity per MIDI pitch for the
                            entire song this sample belongs to.
        """
        song_idx, word_start = self.samples[idx]
        words, midi_filename = self.song_data[song_idx]

        # ---- Text sequence ------------------------------------------------ #
        # Slice the window of words that form the input sequence
        sequence_words = words[word_start : word_start + self.sequence_length]

        # The word immediately after the window is the prediction target
        target_word = words[word_start + self.sequence_length]

        # Look up each word in Word2Vec; use a zero vector for OOV words
        vectors = []
        for word in sequence_words:
            if word in self.wv:
                vectors.append(self.wv[word])          # shape: (300,)
            else:
                vectors.append(np.zeros(300, dtype=np.float32))

        # Stack into a single tensor: (sequence_length, 300)
        text_sequence = torch.tensor(np.array(vectors), dtype=torch.float32)

        # ---- Target ------------------------------------------------------- #
        # Convert the target word to its vocabulary index (0 = <UNK>)
        target_idx = self.word2idx.get(target_word, 0)

        # ---- Melody ------------------------------------------------------- #
        # Get the 128-dim summary vector for this song's MIDI file
        melody_vector = self._extract_melody_vector(midi_filename)  # (128,)

        return {
            'text_sequence': text_sequence,   # (sequence_length, 300)
            'target_word'  : target_idx,      # scalar int
            'melody_vector': melody_vector,   # (128,)
        }

