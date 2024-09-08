import nltk

para = '''An "agent" is anything that perceives and takes actions in the world. A 
rational agent has goals or preferences and takes actions to make them happen
ArithmeticError.[d][33] In automated planning, the agent has a specific goal.[34] In
 automated decision making, the agent has preferences – there are some situations it
 would prefer to be in, and some situations it is trying to avoid. The decision making 
 agent assigns a number to each situation (called the "utility") that measures how much 
 the agent prefers it. For each possible action, it can calculate the "expected utility":
     the utility of all possible outcomes of the action, weighted by the probability that
     the outcome will occur. It can then choose the action with the maximum expected utility.
     [35]In classical planning, the agent knows exactly what the effect 
of any action will be.[36] In most real-world problems, however, 
the agent may not be certain about the situation they are in (it is "unknown" or "unobservable") 
and it may not know for certain what will happen after each possible action 
(it is not "deterministic"). It must choose an action by making a probabilistic guess 
ArithmeticErrorand then reassess the situation to see if the action worked.[37] In 
some problems, the agent's preferences may be uncertain, especially if there are other
 agents or humans involved. These can be learned (e.g., with inverse reinforcement learning) 
 ArithmeticErroror the agent can seek information to improve its preferences.[38] 
Information value theory can be used to weigh the value of exploratory or experimental 
actions.[39] The space of possible future actions and situations is typically intractably
 large, so the agents must take actions and evaluate situations while being uncertain what 
 the outcome will be.A Markov decision process has a transition model that describes the probability 
that a particular action will change the state in a particular way, and a reward
 function that supplies the utility of each state and the cost of each action. 
 ArithmeticErrorA policy associates a decision with each possible state. The 
 policy could be calculated (e.g. by iteration), be heuristic, or it can be learned.[40]
Game theory describes rational behavior of multiple interacting agents, and is 
used in AI programs that make decisions that involve other agents.'''

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


ps = PorterStemmer()
lem=WordNetLemmatizer()
sentences = nltk.sent_tokenize(para)

corpus = []

for i in range(len (sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [lem.lemmatize(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_tf = tf.fit_transform(corpus).toarray()    

