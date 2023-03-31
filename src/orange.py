import Orange
import pandas as pd
from Orange.data import Table
from Orange.classification import CN2Learner, CN2UnorderedLearner
data = Table('titanic')
# data = Table('iris.tab')
learner = CN2Learner()
learner.rule_finder.general_validator.max_rule_length = 5
learner.rule_finder.search_algorithm.beam_width = 10
learner.rule_finder.general_validator.min_covered_examples = 1
classifier1 = learner(data)
df = pd.DataFrame(data.X_df)
df['class'] = data.Y_df

# 'sepal length'
# value=5.843333333333335)
# values = [5.6, 6.2]
# data = Table('iris.tab')
# learner = CN2UnorderedLearner()
#
# # consider up to 10 solution streams at one time
# learner.rule_finder.search_algorithm.beam_width = 10
#
# # continuous value space is constrained to reduce computation time
# learner.rule_finder.search_strategy.bound_continuous = True
#
# # found rules must cover at least 15 examples
# learner.rule_finder.general_validator.min_covered_examples = 15
#
# # found rules must combine at most 2 selectors (conditions)
# learner.rule_finder.general_validator.max_rule_length = 2
#
# classifier2 = learner(data)

for rule in classifier1.rule_list:
    print(rule, rule.curr_class_dist.tolist())
