import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    https://discussions.udacity.com/t/how-to-start-coding-the-selectors/476905
    https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/3

    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #TODO implement model selection based on BIC scores 
        min_bic_score = float("inf")
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            num_data_points = len(self.X)
            feature_len = len(self.X[0])
            num_parameters= n*n + 2*n*feature_len - 1
            model = self.base_model(n)
            try:
                logL = model.score(self.X, self.lengths)
                bic_score = -2*logL + num_parameters*np.log(num_data_points)
                if bic_score < min_bic_score:
                    min_bic_score = bic_score
                    best_model = model
            except:
                pass
        return best_model
          

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # TODO implement model selection based on DIC scores
        best_model = None
        max_dic_score = float("-inf")
        for n in range(self.min_n_components, self.max_n_components+1):
            #compute current work model and score
            current_model = self.base_model(n)
            try:
                logL = current_model.score(self.X, self.lengths)
                remaining_words = [w for w in self.words if w != self.this_word]
                #compute score for remaining words
                M = len(remaining_words)
                Other_sum = 0
                for w in remaining_words:
                    OtherX, Other_lengths = self.hwords[w]
                    try:
                        Other_score = current_model.score(OtherX, Other_lengths)
                        Other_sum += Other_score
                        #compute DIC score for this word
                        dic_score = logL - Other_sum/(M-1)
                        if dic_score > max_dic_score:
                            max_dic_score = dic_score
                            best_model = current_model
                    except:
                        pass
            except:
                pass
        return best_model
       


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        max_logL = float("-inf")
        best_model = None
        num_states = 2
        for n in range(self.min_n_components, self.max_n_components+1):
            # TODO implement model selection using CV
            if len(self.sequences)==1:
                logL = float("-inf")
                model = self.base_model(n)
                try:
                    logL = current_model.score(self.X, self.lengths)
                except:
                    pass
                if logL > max_logL:
                    max_logL = logL
                    best_model = model
                    num_states = n
            else:
                kf = KFold(n_splits=min(3,len(self.sequences)))
                logL_sum = 0
                num_folds = 0
                try:
                    for cv_train_idx, cv_test_idx in kf.split(self.sequences):
                        num_folds += 1
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        try:
                            model = GaussianHMM(n, "diag", 1000, self.random_state, verbose=False).fit(train_X, train_lengths)
                        except:
                            pass
                        try:
                            score = model.score(test_X, test_lengths)
                            logL_sum += score
                        except:
                            pass
                    #get max of avg log Ls for all n_components, and return the component value. 
                    avg_logL = logL_sum/num_folds
                    if avg_logL > max_logL:
                        max_logL = avg_logL
                        best_model = model
                        num_states = n
                except:
                    pass
        #return the model based on the best value of num_states
        return self.base_model(num_states)