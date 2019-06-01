''' Filter TF output
  Takes in TF surp output.
  Removes header junk and <eos> tokens
  Inserts sentids

  usage: python script.py raw_tf.output > filtered_tf.output
'''

import sys

read_phase = 0
sentid = 0
HEADER = True
with open(sys.argv[1],'r') as f:
    for line in f:
        if read_phase < 2 and line[0] == '=':
            read_phase += 1
        elif read_phase == 2:
            sline = line.strip().split()
            if sline[0] == '<eos>':
                sentid += 1
                continue
            else:
                if HEADER:
                    HEADER = False
                    print(sline[0]+' sentid '+sline[1])
                else:
                    print(sline[0]+' '+str(sentid)+' '+sline[1])
