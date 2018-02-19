# coding: utf-8
"""CS585: Assignment 2

In this assignment, you will complete an implementation of
a Hidden Markov Model and use it to fit a part-of-speech tagger.
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os.path
import urllib.request

class HMM:
	def __init__(self, smoothing=0):
		"""
		Construct an HMM model with add-k smoothing.
		Params:
		  smoothing...the add-k smoothing value

		This is DONE.
		"""
		self.smoothing = smoothing
		self.transition_probas ={}
		self.emission_probas = {}
		self.start_probas = {}
		self.states = []

	def fit_transition_probas(self, tags):
		"""
		Estimate the HMM state transition probabilities from the provided data.

		Creates a new instance variable called `transition_probas` that is a
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'N': .1, 'V': .7, 'D': 2},
		 'V': {'N': .3, 'V': .5, 'D': 2},
		 ...
		}
		See test_hmm_fit_transition.

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None
		"""
		###TODO
		pass

		# Unique tokens with count in the sentences
		count_map=defaultdict(int)

		self.states = sorted(set([x for y in tags for x in y]))

		# No end states so we model it such that C(N,*)
		# where C(N, *)$ means "count the number of N tags followed by any other tag"
		for tags_index in range(len(tags)):
			for tag_index in range(len(tags[tags_index])):
				if tag_index != len(tags[tags_index])-1:
					count_map[tags[tags_index][tag_index]]+=1
				else:
					count_map[tags[tags_index][tag_index]]+=0

		map_dic = defaultdict(lambda : defaultdict(int))

		# Tag1 :{Tag2: Count} No of times tag2 follows tag1 in our lists of lists of tags
		for tag_l in tags:
			for i in range(len(tag_l)):
				if i<len(tag_l)-1:
					map_dic[tag_l[i]][tag_l[i+1]]+=1


		for tag in count_map:
			self.transition_probas[tag] = defaultdict(int)

		for tag_1 in count_map:
			for tag_2 in count_map:
				if  self.smoothing != 0:
					self.transition_probas[tag_2][tag_1] = (map_dic[tag_2][tag_1]+self.smoothing)/(count_map[tag_2]+(len(count_map)*self.smoothing))
				else:
					if count_map[tag_2] != 0:
						self.transition_probas[tag_2][tag_1] = (map_dic[tag_2][tag_1])/(count_map[tag_2])
					else:
						self.transition_probas[tag_2][tag_1] = 0.0

	def fit_emission_probas(self, sentences, tags):
		"""
		Estimate the HMM emission probabilities from the provided data.

		Creates a new instance variable called `emission_probas` that is a
		dict from a string ('state') to a dict from string to float. E.g.
		{'N': {'dog': .1, 'cat': .7, 'mouse': 2},
		 'V': {'run': .3, 'go': .5, 'jump': 2},
		 ...
		}

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_emission.
		"""
		###TODO
		pass

		unique_words_forsmooth = set([x for y in sentences for x in y])

		count_map=defaultdict(int)

		for tag_s in tags:
			for tag in tag_s:
				count_map[tag]+=1

		# {Word : {tag: count(word,tag)}}
		word_tag_map = defaultdict(lambda : defaultdict(float))
		for i in range(len(sentences)):
			for j in range(len(sentences[i])):
				word_tag_map[sentences[i][j]][tags[i][j]] += 1

		for key in word_tag_map:
			for tag in count_map:
				if tag not in word_tag_map[key]:
					word_tag_map[key][tag] = 0.0

		self.emission_probas = defaultdict(lambda : defaultdict())

		for word in word_tag_map:
			for tag in word_tag_map[word]:
				self.emission_probas[tag][word] = (word_tag_map[word][tag] + self.smoothing)/(count_map[tag] + (len(unique_words_forsmooth)*self.smoothing))

		# print(self.emission_probas)

	def fit_start_probas(self, tags):
		"""
		Estimate the HMM start probabilities form the provided data.

		Creates a new instance variable called `start_probas` that is a
		dict from string (state) to float indicating the probability of that
		state starting a sentence. E.g.:
		{
			'N': .4,
			'D': .5,
			'V': .1
		}

		Params:
		  tags...a list of lists of strings representing the tags for one sentence.
		Returns:
			None

		See test_hmm_fit_start
		"""
		###TODO
		pass

		all_tags = set([x for y in tags for x in y])

		tag_start_map = defaultdict(int)
		for tag in all_tags:
			tag_start_map[tag] = 0

		for tag_l in tags:
			# print(tag_l)
			if tag_l[0] in tag_start_map:
				tag_start_map[tag_l[0]]+=1

		self.start_probas = defaultdict(float)

		for tag in tag_start_map:
			self.start_probas[tag] = (tag_start_map[tag] + self.smoothing)/(len(tags) + (len(all_tags) * self.smoothing))


	def fit(self, sentences, tags):
		"""
		Fit the parameters of this HMM from the provided data.

		Params:
		  sentences...a list of lists of strings, representing the tokens in each sentence.
		  tags........a list of lists of strings, representing the tags for one sentence.
		Returns:
			None

		DONE. This just calls the three fit_ methods above.
		"""
		self.fit_transition_probas(tags)
		self.fit_emission_probas(sentences, tags)
		self.fit_start_probas(tags)


	def viterbi(self, sentence):
		"""
		Perform Viterbi search to identify the most probable set of hidden states for
		the provided input sentence.

		Params:
		  sentence...a lists of strings, representing the tokens in a single sentence.

		Returns:
		  path....a list of strings indicating the most probable path of POS tags for
		  		  this sentence.
		  proba...a float indicating the probability of this path.
		"""
		###TODO
		pass

		#Step 1 Initialization:
		viterbi_matrix = np.ndarray(shape=(len(self.states)+2,len(sentence)),dtype=float)

		for state_index in range(len(self.states)):
			viterbi_matrix[state_index,0] = self.start_probas[self.states[state_index]] * self.emission_probas[self.states[state_index]][sentence[0]]

		#Step 2 Recursion:
		for word_index in range(1,len(sentence)):
			for state_index in range(len(self.states)):
				max_v = max([viterbi_matrix[s_dash,word_index-1] *(self.transition_probas[self.states[s_dash]][self.states[state_index]])*(self.emission_probas[self.states[state_index]][sentence[word_index]]) for s_dash in range(len(self.states))])
				viterbi_matrix[state_index,word_index] = max_v

		# Step 3 Termination:
		# Final State Index and final Observation T'Index , ie last word in our sentence.
		final_max = max([viterbi_matrix[s_dash,len(sentence)-1] for s_dash in range(len(self.states))])
		viterbi_matrix[len(self.states)+1,len(sentence)-1] = final_max

		#Step 4 BackTrackPath:
		result_path = []

		for x in range(len(sentence)):
			max_val_state = 0
			temp = []
			temp_index = []
			for y in range(len(self.states)):
				temp.append(viterbi_matrix[y,x])
				temp_index.append(y)
			max_val_state = max(temp)
			result_path.append(self.states[temp_index[temp.index(max_val_state)]])

		return result_path,final_max

def read_labeled_data(filename):
	"""
	Read in the training data, consisting of sentences and their POS tags.

	Each line has the format:
	<token> <tag>

	New sentences are indicated by a newline. E.g. two sentences may look like this:
	<token1> <tag1>
	<token2> <tag2>

	<token1> <tag1>
	<token2> <tag2>
	...

	See data.txt for example data.

	Params:
	  filename...a string storing the path to the labeled data file.
	Returns:
	  sentences...a list of lists of strings, representing the tokens in each sentence.
	  tags........a lists of lists of strings, representing the POS tags for each sentence.
	"""
	###TODO
	pass

	with open(filename,'r') as myfile:
		data = myfile.read()
	myfile.close()

	sentences = data.split("\n\n")

	sentencex = []
	tags = []
	for sentence in sentences:
		words =sentence.split("\n")
		tokens = []
		tag = []
		for token_tag in words:
			temp = token_tag.strip().split(" ")
			if len(temp)>1:
				tokens.append(temp[0])
				tag.append(temp[1])

		if len(tokens)>1 and len(tag)>1:
			sentencex.append(tokens)
			tags.append(tag)

	return sentencex,tags


def download_data():
    """ Download labeled data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/ty7cclxiob3ajog/data.txt?dl=1'
    urllib.request.urlretrieve(url, 'data.txt')

if __name__ == '__main__':
	"""
	Read the labeled data, fit an HMM, and predict the POS tags for the sentence
	'Look at what happened'

	DONE - please do not modify this method.

	The expected output is below. (Note that the probability may differ slightly due
	to different computing environments.)

	$ python3 a2.py
	model has 34 states
        ['$', "''", ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'TO', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WRB', '``']
	predicted parts of speech for the sentence ['Look', 'at', 'what', 'happened']
	(['VB', 'IN', 'WP', 'VBD'], 2.751820088075314e-10)
	"""
	fname = 'data.txt'
	if not os.path.isfile(fname):
		download_data()
	sentences, tags = read_labeled_data(fname)

	model = HMM(.001)
	model.fit(sentences, tags)
	print('model has %d states' % len(model.states))
	print(model.states)
	sentence = ['Look', 'at', 'what', 'happened']
	print('predicted parts of speech for the sentence %s' % str(sentence))
	print(model.viterbi(sentence))
