import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
  """ Recognize test word sequences from word models set

 :param models: dict of trained models
     {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
 :param test_set: SinglesData object
 :return: (list, list)  as probabilities, guesses
     both lists are ordered by the test set word_id
     probabilities is a list of dictionaries where each key a word and value is Log Liklihood
         [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
          ]
     guesses is a list of the best guess words ordered by the test set word_id
         ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
 """

  warnings.filterwarnings("ignore", category=DeprecationWarning)
  probabilities = []
  guesses = []
  g_dict = {}
  # TODO implement the recognizer
  for i in range(len(test_set.df)):
    X_test_item, lengths_test_item = test_set.get_item_Xlengths(i)
    max_logL = float("-inf")
    p_dict = {}
    for word in models:
      model_to_test_on = models[word]
      try:
        logL = model_to_test_on.score(X_test_item, lengths_test_item)
      except:
        pass
      p_dict[word] = logL
      if logL > max_logL:
        max_logL = logL
        best_match = word
    guesses.append(best_match)
    probabilities.append(p_dict)
  return probabilities, guesses
