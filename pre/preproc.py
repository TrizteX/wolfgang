import glob
import numpy
from music21 import chord, converter, instrument, note


class Preprocessing():
    def __init__(self):
        self.notes = self.get_notes()
        self.len = len(set(self.notes))

        self.input_seq, self.out_seq  = self.prepare_sequences(self.notes, self.len)
        
    def prepare_sequences(self, notes, n_vocab):
        '''
        Exctracted Notes To A Numpy 
        Array
        notes -> from get_notes
        vocab -> len of notes returned 
        '''
        sequence_length = 100

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

        network_input = []
        network_output = []

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])

        n_patterns = len(network_input)

        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab)

        network_output = numpy.array(network_output)
        

        return (network_input, network_output)


    def get_notes(self):
        '''
        Notes And Chord 
        From The midi

        '''
        notes = []

        for file in glob.glob("midi_songs/*.mid"): ##change
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except: # file has notes in a flat structure
                notes_to_parse = midi.flat.notesAndRests

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                elif isinstance(element, note.Rest): #ADDED
                    notes.append(element.name) #ADDED



        return notes

   