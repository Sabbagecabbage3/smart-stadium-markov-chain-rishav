import pandas as pd
import numpy as np

class Markov():
    def __init__(self) -> None:
        self.plays = pd.read_csv('allplays.csv')
        self.plays = self.plays[self.plays['type'].notna()]
        self.plays = self.plays[self.plays.type != '#'] #take drive start plays
        self.plays = self.plays[self.plays.type != 'E'] #take penalties
    def createDownMarkov(self):
        self.down_matrix = np.zeros((4,5))
        # self.plays = pd.read_csv('allplays.csv')
        # self.plays = self.plays[self.plays['type'].notna()]
        # self.plays = self.plays[self.plays.type != '#'] #take drive start plays
        # self.plays = self.plays[self.plays.type != 'E'] #take penalties
        plays = self.plays
        play_counts = {'1': 0, '2': 0, '3': 0,'4': 0}
        curr_down,next_down = 1,2
        for x in range(plays.shape[0]-1):
            curr_down = plays['context'].iloc[x].split(',')[1] #context format is H,1,10,H35
            next_down = plays['context'].iloc[x+1].split(',')[1] #context format is H,1,10,H35
            if plays['playID'].iloc[x].split(',')[0] == plays['playID'].iloc[x+1].split(',')[0]: #if same drive
                play_counts[curr_down] = play_counts[curr_down] + 1
                self.down_matrix[int(curr_down)-1, int(next_down)-1] += 1
            
        for p in range(4):
            self.down_matrix[p] /= play_counts[str(p+1)]
        self.down_matrix[:,4] = self.down_matrix.sum(axis = 1)
        # self.down_matrix.set_printoptions(suppress=True)
        np.set_printoptions(suppress=True) #suppresses scientific notation
        print(self.plays.shape)
        print(play_counts)
        print(self.down_matrix)
        down_labels = list(play_counts.keys())
        down_labels.append('Total')
        df = pd.DataFrame(self.down_matrix, columns=down_labels, index=play_counts.keys())
        df.to_csv('Down.csv')
        return
    def createDownTypeMarkov(self):
        # self.down_type_matrix = np.zeros((28,29)) #4 downs * 7 play types + an extra column for sums
        self.down_type_matrix = np.zeros((28,36))
        self.plays = pd.read_csv('allplays.csv')
        self.plays = self.plays[self.plays['type'].notna()]
        self.plays = self.plays[self.plays['type'].notna()]
        self.plays = self.plays[self.plays.type != '#'] #take drive start plays
        self.plays = self.plays[self.plays.type != 'E'] #take penalties
        plays = self.plays

        playType_counts = {'R':0, 'P':0, 'X':0, 'F':0, 'U':0, 'K':0, 'O':0}
        playType_ID =  {'R':0, 'P':1, 'X':2, 'F':3, 'U':4, 'K':5, 'O':6}
        for x in range(plays.shape[0]-1):
            curr_down = plays['context'].iloc[x].split(',')[1] #context format is H,1,10,H35
            next_down = plays['context'].iloc[x+1].split(',')[1] #context format is H,1,10,H35
            curr_type = plays['type'].iloc[x] if plays['type'].iloc[x] in playType_ID.keys() else 'O'
            next_type = plays['type'].iloc[x+1] if plays['type'].iloc[x+1] in playType_ID.keys() else 'O'
            curr_index = (int(curr_down)-1) * (len(playType_ID.keys())) + playType_ID[curr_type] # indices are grouped by down and type added as offset
            next_index = (int(next_down)-1) * (len(playType_ID.keys())) + playType_ID[next_type]
            if plays['playID'].iloc[x].split(',')[0] == plays['playID'].iloc[x+1].split(',')[0]: #if same drive
                # print(True)
                # if curr_down == '4' and curr_type == 'U': #edge case: safety on punt, so had to kickoff on same drive
                #     print(curr_index, next_index)
                #     print(plays.iloc[x])
                #     print(plays.iloc[x+1])
                #     print(plays['playID'].iloc[x].split(',')[0],plays['playID'].iloc[x+1].split(',')[0])
                self.down_type_matrix[curr_index,next_index] += 1
            else:
                if int(plays['playID'].iloc[x].split(',')[0]) + 1 == int(plays['playID'].iloc[x+1].split(',')[0]): #if last play was punt/kickoff leading to next drive
                    if curr_type == 'U' or curr_type == 'K':
                        self.down_type_matrix[curr_index,next_index] += 1
            # if curr_down == '4' and curr_type == 'U' and next_type == 'K':
            #     print(plays.iloc[x])
            #     print(plays.iloc[x+1])
        np.set_printoptions(suppress=True) #suppresses scientific notation
        
        self.down_type_matrix[:,-8] = np.sum(self.down_type_matrix, axis=1)
        for s in range(self.down_type_matrix.shape[0]):
            if self.down_type_matrix[s,-8] != 0:
                self.down_type_matrix[s,:] /= self.down_type_matrix[s,-8]
        for s in range(self.down_type_matrix.shape[0]):
            if self.down_type_matrix[s,-8] != 0:
                # self.down_type_matrix[s,:] /= self.down_type_matrix[s,-8]
                self.down_type_matrix[s,-7] = self.down_type_matrix[s,0] + self.down_type_matrix[s,7] + self.down_type_matrix[s,14] + self.down_type_matrix[s,21]
                self.down_type_matrix[s,-6] = self.down_type_matrix[s,1] + self.down_type_matrix[s,8] + self.down_type_matrix[s,15] + self.down_type_matrix[s,22]
                self.down_type_matrix[s,-5] = self.down_type_matrix[s,2] + self.down_type_matrix[s,9] + self.down_type_matrix[s,16] + self.down_type_matrix[s,23]
                self.down_type_matrix[s,-4] = self.down_type_matrix[s,3] + self.down_type_matrix[s,10] + self.down_type_matrix[s,17] + self.down_type_matrix[s,24]
                self.down_type_matrix[s,-3] = self.down_type_matrix[s,4] + self.down_type_matrix[s,11] + self.down_type_matrix[s,18] + self.down_type_matrix[s,25]
                self.down_type_matrix[s,-2] = self.down_type_matrix[s,5] + self.down_type_matrix[s,12] + self.down_type_matrix[s,19] + self.down_type_matrix[s,26]
                self.down_type_matrix[s,-1] = self.down_type_matrix[s,6] + self.down_type_matrix[s,13] + self.down_type_matrix[s,20] + self.down_type_matrix[s,27]
        down_type_labels = []
        for d in range(1,5): #downs 1-4
            for t in range(len(playType_ID.keys())):
                down_type_labels.append(str(d) + list(playType_ID.keys())[t])
        # print(down_type_labels)
        down_type_labels.append('Total')
        down_type_labels.extend(list(playType_ID.keys()))
        df = pd.DataFrame(self.down_type_matrix, columns = down_type_labels, index=down_type_labels[:-8])
        df.to_csv('DownType.csv')
        print(df)
        # print(self.down_type_matrix)
        #print(self.down_type_matrix)
        
        return
    def createDownTypeFPMarkov(self):
        self.down_type_fp_matrix = np.zeros((280,288)) #4 downs * 7 types * 10 FieldPosition buckets + extra sum column
        plays = self.plays
        playType_ID =  {'R':0, 'P':1, 'X':2, 'F':3, 'U':4, 'K':5, 'O':6}
        fp_ID = {'H': {x:x for x in range(5)}, 'V': {x:y for (x,y) in zip ([x for x in range(4,-1,-1)], [y for y in range(5,10)])}} #check if on own side (H) or other side (V), then sort into 0-4 based on fp of 0-9, 10-19, 20s, 30s,40s, 40s, 30s, etc
        for x in range(plays.shape[0]-1):
            curr_down = plays['context'].iloc[x].split(',')[1] #context format is H,1,10,H35
            next_down = plays['context'].iloc[x+1].split(',')[1] #context format is H,1,10,H35
            curr_type = plays['type'].iloc[x] if plays['type'].iloc[x] in playType_ID.keys() else 'O'
            next_type = plays['type'].iloc[x+1] if plays['type'].iloc[x+1] in playType_ID.keys() else 'O'
            curr_possessor = plays['context'].iloc[x].split(',')[0]
            curr_side = 'H' if plays['context'].iloc[x].split(',')[3][0] == curr_possessor else 'V'
            curr_fp = int(plays['context'].iloc[x].split(',')[3][1:]) if int(plays['context'].iloc[x].split(',')[3][1:]) < 50 else 49
            next_possessor = plays['context'].iloc[x+1].split(',')[0]
            next_side = 'H' if plays['context'].iloc[x+1].split(',')[3][0] == next_possessor else 'V'
            next_fp = int(plays['context'].iloc[x+1].split(',')[3][1:]) if int(plays['context'].iloc[x+1].split(',')[3][1:]) < 50 else 49
            # curr_index = (int(curr_down)-1) * (playType_ID[curr_type] * len(playType_ID.keys())) + fp_ID[curr_side][curr_fp//10] # indices are grouped by down, then type, then fp added as offset
            # next_index = (int(next_down)-1) * (playType_ID[next_type] * len(playType_ID.keys())) + fp_ID[next_side][next_fp//10]
            t_len, f_len = len(playType_ID.keys()), 10
            curr_index = ((int(curr_down) - 1) *t_len*f_len) + ((playType_ID[curr_type])*f_len) + ((fp_ID[curr_side][curr_fp//10]))
            next_index = ((int(next_down) - 1) *t_len*f_len) + ((playType_ID[next_type])*f_len) + ((fp_ID[next_side][next_fp//10]))
            if plays['playID'].iloc[x].split(',')[0] == plays['playID'].iloc[x+1].split(',')[0]: #if same drive
                self.down_type_fp_matrix[curr_index,next_index] += 1
            else:
                if int(plays['playID'].iloc[x].split(',')[0]) + 1 == int(plays['playID'].iloc[x+1].split(',')[0]): #if last play was punt/kickoff leading to next drive
                    if curr_type == 'U' or curr_type == 'K':
                        self.down_type_fp_matrix[curr_index,next_index] += 1
        np.set_printoptions(suppress=True) #suppresses scientific notation
        self.down_type_fp_matrix[:, -8] = np.sum(self.down_type_fp_matrix, axis=1)
        for s in range(self.down_type_fp_matrix.shape[0]):
            if self.down_type_fp_matrix[s,-8] != 0:
                self.down_type_fp_matrix[s,:] /= self.down_type_fp_matrix[s,-8]
        play_type_indices = [x for x in range(0,211,70)] + [x for x in range(1,212,70)] + [x for x in range(2,213,70)] + [x for x in range(3,214,70)] + [x for x in range(4,215,70)] + [x for x in range(5,216,70)] + [x for x in range(6,217,70)] + [x for x in range(7,218,70)] + [x for x in range(8,219,70)] + [x for x in range(9,220,70)]
        for s in range(self.down_type_fp_matrix.shape[0]):
            if self.down_type_fp_matrix[s,-8] != 0:
                # self.down_type_fp_matrix[s,:] /= self.down_type_fp_matrix[s,-8]
                curr_type_indices = np.array(play_type_indices)
                play_type_counter = -7
                while play_type_counter < 0:
                    p_count = 0
                    for x in curr_type_indices:    
                        p_count += self.down_type_fp_matrix[s,x]
                    self.down_type_fp_matrix[s,play_type_counter] = p_count
                    curr_type_indices += 10
                    play_type_counter += 1
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-6] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-5] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-4] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-3] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-2] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
            # for x in curr_type_indices:
            #     self.down_type_fp_matrix[s,-1] += self.down_type_fp_matrix[s,x]
            # curr_type_indices += 10
        down_type_fp_labels = []
        for d in range(1,5): #downs 1-4
            for t in range(len(playType_ID.keys())):
                for fp in fp_ID['H'].keys():
                    down_type_fp_labels.append(str(d) + list(playType_ID.keys())[t] + "@H" + str(fp))
                for fp in fp_ID['V'].keys():
                    down_type_fp_labels.append(str(d) + list(playType_ID.keys())[t] + "@V" + str(fp))
        # print(down_type_fp_labels)
        down_type_fp_labels.append('Total')
        down_type_fp_labels.extend(list(playType_ID.keys()))
        df = pd.DataFrame(self.down_type_fp_matrix, columns = down_type_fp_labels, index=down_type_fp_labels[:-8])
        df.to_csv('DownTypeFP.csv')
        print(df)
    def createDownTypeFPDistMarkov(self):
        self.down_type_fp_dist_matrix = np.zeros((1680,1688)) #4 downs * 7 types * 10 FieldPosition buckets * 6 distance buckets + 7 play type aggregates + extra sum column
        plays = self.plays
        playType_ID =  {'R':0, 'P':1, 'X':2, 'F':3, 'U':4, 'K':5, 'O':6}
        fp_ID = {'H': {x:x for x in range(5)}, 'V': {x:y for (x,y) in zip ([x for x in range(4,-1,-1)], [y for y in range(5,10)])}} #check if on own side (H) or other side (V), then sort into 0-4 based on fp of 0-9, 10-19, 20s, 30s,40s, 40s, 30s, etc
        dist_ID = {'1-3':0, '4-6': 1, '7-9':2, '10-14': 3, '15-19': 4, '20-99': 5}
        for x in range(plays.shape[0]-1):
            curr_down = plays['context'].iloc[x].split(',')[1] #context format is H,1,10,H35
            next_down = plays['context'].iloc[x+1].split(',')[1] #context format is H,1,10,H35
            curr_type = plays['type'].iloc[x] if plays['type'].iloc[x] in playType_ID.keys() else 'O'
            next_type = plays['type'].iloc[x+1] if plays['type'].iloc[x+1] in playType_ID.keys() else 'O'
            curr_possessor = plays['context'].iloc[x].split(',')[0]
            curr_side = 'H' if plays['context'].iloc[x].split(',')[3][0] == curr_possessor else 'V'
            curr_fp = int(plays['context'].iloc[x].split(',')[3][1:]) if int(plays['context'].iloc[x].split(',')[3][1:]) < 50 else 49
            next_possessor = plays['context'].iloc[x+1].split(',')[0]
            next_side = 'H' if plays['context'].iloc[x+1].split(',')[3][0] == next_possessor else 'V'
            next_fp = int(plays['context'].iloc[x+1].split(',')[3][1:]) if int(plays['context'].iloc[x+1].split(',')[3][1:]) < 50 else 49
            curr_dist = int(plays['context'].iloc[x].split(',')[2])
            curr_dist_id = 0 if curr_dist <= 3 else 1 if curr_dist <=6 else 2 if curr_dist <= 9 else 3 if curr_dist<=14 else 4 if curr_dist <=19 else 5
            next_dist = int(plays['context'].iloc[x+1].split(',')[2])
            next_dist_id = 0 if next_dist <= 3 else 1 if next_dist <=6 else 2 if next_dist <= 9 else 3 if next_dist<=14 else 4 if next_dist <=19 else 5
            t_len, f_len, di_len = len(playType_ID.keys()), 10, len(dist_ID.keys())
            #curr_index = (int(curr_down)-1) * (playType_ID[curr_type] * len(playType_ID.keys())) * (fp_ID[curr_side][curr_fp//10]) + curr_dist_id # indices are grouped by down, then type, then fp, then dist added as offset
            #next_index = (int(next_down)-1) * (playType_ID[next_type] * len(playType_ID.keys())) * (fp_ID[next_side][next_fp//10]) + next_dist_id
            curr_index = ((int(curr_down) - 1) *t_len*f_len*di_len) + ((playType_ID[curr_type])*f_len*di_len) + ((fp_ID[curr_side][curr_fp//10])*di_len) + curr_dist_id
            next_index = ((int(next_down) - 1) *t_len*f_len*di_len) + ((playType_ID[next_type])*f_len*di_len) + ((fp_ID[next_side][next_fp//10])*di_len) + next_dist_id
            # if (curr_index == 253 or curr_index == 254):
            #     print(plays.iloc[x-1])
            #     print(plays.iloc[x])
            # if curr_down != '4' and curr_type == 'U': #incorrectly recorded next play as 3rd down: 11996,"3,11,18",P,,"PASS:10,S,TM,V38 SACK:49 PEN:V,IG,A,10,V38,N","V,3,15,H44","PATTERSON, P. sacked for loss of 18 yards to the SYRACUSE38 (Wilkinson, G.), PENALTY SYRACUSE intentional grounding (PATTERSON, P.) 0 yards to the SYRACUSE38."
            #     print(plays.iloc[x-1])
            #     print(plays.iloc[x])
            #     print(plays.iloc[x+1])
            # # print(curr_index, next_index)
            # print(curr_down, curr_type, curr_side, curr_fp, curr_dist)
            if plays['playID'].iloc[x].split(',')[0] == plays['playID'].iloc[x+1].split(',')[0]: #if same drive
                self.down_type_fp_dist_matrix[curr_index,next_index] += 1
            else:
                if int(plays['playID'].iloc[x].split(',')[0]) + 1 == int(plays['playID'].iloc[x+1].split(',')[0]): #if last play was punt/kickoff leading to next drive
                    if curr_type == 'U' or curr_type == 'K':
                        self.down_type_fp_dist_matrix[curr_index,next_index] += 1
        np.set_printoptions(suppress=True) #suppresses scientific notation
        self.down_type_fp_dist_matrix[:, -8] = np.sum(self.down_type_fp_dist_matrix, axis=1)
        for s in range(self.down_type_fp_dist_matrix.shape[0]):
            if self.down_type_fp_dist_matrix[s,-8] != 0:
                self.down_type_fp_dist_matrix[s,:] /= self.down_type_fp_dist_matrix[s,-8] #normalize probabilities for each row
        play_type_indices = [x for x in range(0,60)] + [x for x in range(420,480)] + [x for x in range(840, 900)] + [x for x in range(1260,1320)]
        for s in range(self.down_type_fp_dist_matrix.shape[0]): #aggregate values for each state
            if self.down_type_fp_dist_matrix[s,-8] != 0:
                # self.down_type_fp_dist_matrix[s,:] /= self.down_type_fp_dist_matrix[s,-8]
                curr_type_indices = np.array(play_type_indices)
                play_type_counter = -7
                while play_type_counter < 0:
                    p_count = 0
                    for x in curr_type_indices:    
                        p_count += self.down_type_fp_dist_matrix[s,x]
                    self.down_type_fp_dist_matrix[s,play_type_counter] = p_count
                    curr_type_indices += 60
                    play_type_counter += 1

        down_type_fp_dist_labels = []
        for d in range(1,5): #downs 1-4
            for t in range(len(playType_ID.keys())):
                for fp in fp_ID['H'].keys():
                    for dist in range(len(dist_ID.keys())):
                        down_type_fp_dist_labels.append(str(d) + "&" + list(dist_ID.keys())[dist] + " " + list(playType_ID.keys())[t] + "@H" + str(fp) + "0")
                for fp in fp_ID['V'].keys():
                    for dist in range(len(dist_ID.keys())):
                        down_type_fp_dist_labels.append(str(d) + "&" + list(dist_ID.keys())[dist] + " " + list(playType_ID.keys())[t] + "@V" + str(fp) + "0")
        # print(down_type_fp_labels)
        down_type_fp_dist_labels.append('Total')
        down_type_fp_dist_labels.extend(list(playType_ID.keys()))
        df = pd.DataFrame(self.down_type_fp_dist_matrix, columns = down_type_fp_dist_labels, index=down_type_fp_dist_labels[:-8])
        
        df.to_csv('DownTypeFPDist.csv')
        #print(df)
        return
# Markov().createDownMarkov()
#Markov().createDownTypeMarkov()
Markov().createDownTypeFPMarkov()
# Markov().createDownTypeFPDistMarkov()




