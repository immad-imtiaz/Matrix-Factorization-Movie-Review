import pandas as pd
import numpy as np
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import graphlab as gl
from graphlab import SFrame
import matplotlib.pyplot as plt


# Read Training Rating Data
# Total Rows: 2038130
train_rating_df = pd.read_csv('train_rating.csv')
# Read Testing Rating Data
# Total Rows: 158024
test_rating_df = pd.read_csv('test_rating.txt', sep=",")
test_rating_for_answer_join = test_rating_df[['business_id', 'user_id', 'test_id']]



# Read Sampled Submission
sampled_submission = pd.read_csv('sampled_submission.csv', sep=",")
# Get Total Business and Users in Testing Data
n_users = test_rating_df.user_id.unique()
n_business = test_rating_df.business_id.unique()
# Get Total Business and Users in Training Data
test_n_users = train_rating_df.user_id.unique().shape[0]
test_n_business = train_rating_df.business_id.unique().shape[0]
max_items_in_utility_matrix = n_users.size*n_business.size

# Data too large for numpy
# # Creating User Item Matrix for training
# train_matrix = np.zeros((train_n_users, train_n_business))
# # Creating User Item Matrix for testing
# test_matrix = np.zeros((test_n_users, test_n_business))

print('--Training Data--')
print('Users: %s' % n_users.size)
print('Business:%s' % n_business.size)
print('Matrix Size: %s * %s' % train_rating_df.shape)
print('Expected Utility Matrix Items: %s' % max_items_in_utility_matrix)
print('Matrix Actual Size: %s' % train_rating_df.shape[0])
data_percentage = train_rating_df.shape[0]/float(max_items_in_utility_matrix)
print('Matrix Data Sparsity: %s' % ((1-data_percentage)*100))
print('Total User Pair: %s' % ((n_users.size*(n_users.size-1))/2))
print('Total Business Pair: %s' % ((n_business.size*(n_business.size-1))/2))
train_test_size = int((train_rating_df.shape[0])*0.1)
print('Training Testing Data Size 10 percent: %s' % str(train_test_size))

# --Training Data--
# Users: 75541
# Business:50017
# Matrix Size: 2038130 * 5
# Expected Utility Matrix Items: 3778334197
# Matrix Actual Size: 2038130
# Matrix Data Sparsity: 99.9460574451
# Total User Pair: 2853183570
# Total Business Pair: 1250825136


def data_pre_processing(df, train=False, add_columns=False):
    if add_columns:
        # Getting User Mean df
        user_rating_mean_df = df[['user_id', 'rating']]
        user_rating_mean_df = user_rating_mean_df.groupby('user_id').mean().add_prefix('mean_user_')\
            .reset_index()

        # Getting Item mean df
        item_rating_mean_df = df[['business_id', 'rating']]
        item_rating_mean_df = item_rating_mean_df.groupby('business_id').mean().add_prefix('mean_business_')\
            .reset_index()

        # Now join the data original data
        df = pd.merge(df, item_rating_mean_df,
                      left_on=['business_id'], right_on=['business_id'], how='inner')
        df = pd.merge(df, user_rating_mean_df, how='inner', left_on=['user_id'], right_on=['user_id'])

        # In order to normalize a utility matrix we subtract the average rating of each user
        # from the rating converting low rating as negative numbers and high rating as positive numbers
        df['rating_normalized_by_user'] = df['rating'] - df['mean_user_rating']
        df['rating_normalized_by_item'] = df['rating'] - df['mean_business_rating']
    df = df.rename(columns={'business_id': 'item_id'}, inplace=False)
    if train:
        del df['train_id']
    if not train:
        del df['test_id']
    df['date'] = pd.to_datetime(df['date']).astype(np.int64)
    return df

train_rating_df = data_pre_processing(train_rating_df, train=True)
test_rating_df = data_pre_processing(test_rating_df)

sf_train = gl.SFrame(data=train_rating_df)
sf_test = gl.SFrame(data=test_rating_df)
sf_answer_join = gl.SFrame(data=test_rating_for_answer_join)

train, test = gl.recommender.util.random_split_by_user(sf_train, max_num_users=train_test_size)


# With Extra added columns and regularization of 0.1
# Optimization Complete: Maximum number of passes through the data reached.
# Computing final objective value and training RMSE.
#        Final objective value: 0.723868
#        Final training RMSE: 0.383992

#Creating model
# m = gl.ranking_factorization_recommender.create(train, user_id='user_id',
#                                                 ranking_regularization=0.1,
#                                                 item_id='item_id', target='rating')

m = gl.ranking_factorization_recommender.create(train, user_id='user_id',
                                                ranking_regularization=1,
                                                item_id='item_id', target='rating')

# model = gl.load_model('ranking_factorization_recommender_.1')
# predict = model.predict(gl.SFrame(test_rating_df))



# Predicting Score
# prediction = m.evaluate(sf_test)



# train_shuffled = gl.toolkits.cross_validation.shuffle(sf_train, random_seed=1)
#
#
# folds = gl.cross_validation.KFold(sf_train, 5)
# for train, valid in folds:
#     m = gl.ranking_factorization_recommender.create(sf_train, user_id='user_id',
#                                                     ranking_regularization=0.1,
#                                                     item_id='business_id', target='rating')
#     print m.evaluate(valid)



# With Extra added columns and regularization of 0.12
# Optimization Complete: Maximum number of passes through the data reached.
# Computing final objective value and training RMSE.
#        Final objective value: 0.839013
#        Final training RMSE: 0.475723



# With Out Extra Added Columns and no regularization
# Optimization Complete: Maximum number of passes through the data reached.
# Computing final objective value and training RMSE.
# Final objective value: 1.22228
# Final training RMSE: 0.533115


# With Extra Added Columns
# Optimization Complete: Maximum number of passes through the data reached (hard limit).
# Computing final objective value and training RMSE.
#        Final objective value: 3.85757
#        Final training RMSE: 1.3958




# # Load Training meta data from json
# def load_json_from_file(filename):
#     """
#     Load JSON from a file.
#     @input  filename  Name of the file to be read.
#     @returns Output SFrame
#     """
#     # # Read the entire file into a SFrame with one row
#     sf = gl.SFrame.read_csv(filename, delimiter='', header=False)
#     # sf = sf.stack('X1', new_column_type=[str, int])
#     # # The dictionary can be unpacked to generate the individual columns.
#     # sf = sf.unpack('X1', column_name_prefix='')
#     return sf
# train_meta_sf = load_json_from_file('train_review.json')



# Since only 1% of pairs exist we use triples method for storing utility matrix
# Creating a Utility Matrix using triples method

# triple_index = pd.MultiIndex(levels=[[], []],
#                              labels=[[], []],
#                              names=[u'business_id', u'user_id'])
#
# utility_matrix_normalized = train_rating_df.set_index(['user_id', 'business_id'])
# utility_matrix_normalized = utility_matrix_normalized[['rating']]

# In order to normalize a utility matrix we subtract the average rating of each user
# from the rating converting low rating as negative numbers and high rating as positive numbers


#
#
#
# train_rating_df.apply(add_row_to_utility_matrix, axis=1)





































