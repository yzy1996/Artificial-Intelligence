'''using sklearn
'''
from sklearn import preprocessing

# enc = preprocessing.OrdinalEncoder() # 整数编码
# enc = preprocessing.OneHotEncoder() # 独热编码

genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']

enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])

integer_codes = enc.fit_transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()
print(integer_codes)
# [[1. 0. 0. 0. 0. 1. 0. 0. 0. 1.]
# [0. 1. 0. 0. 1. 0. 0. 0. 0. 1.]]

original_representation = enc.inverse_transform([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
print(original_representation)
# [['female' 'from US' 'uses Safari']]
