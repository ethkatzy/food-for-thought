from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pd.options.mode.chained_assignment = None

recipesDf = pd.read_csv('recipes.csv')
interactionsDf = pd.read_csv('interactions.csv')
#Single reviewer ID: 361457


def convertStringToList(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


mlbTags = MultiLabelBinarizer()
mlbIngredients = MultiLabelBinarizer()
recipesDf['ingredients'] = recipesDf['ingredients'].apply(convertStringToList)
recipesDf['tags'] = recipesDf['tags'].apply(convertStringToList)
recipesDf['tags_vector'] = list(mlbTags.fit_transform(recipesDf['tags']))
recipesDf['ingredients_vector'] = list(mlbTags.fit_transform(recipesDf['ingredients']))


def recommendHybrid(userId, interactionsDf, recipesDf, tagsMapping, topN=5):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactionsDf[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainSet)
    ratedRecipes = interactionsDf[interactionsDf['user_id'] == userId]['recipe_id'].tolist()
    if len(ratedRecipes) >= 2:
        unratedRecipes = recipesDf[~recipesDf['recipe_id'].isin(ratedRecipes)]
        predictions = []
        for recipeId in unratedRecipes['recipe_id']:
            pred = model.predict(userId, recipeId)
            predictions.append((recipeId, pred.est))
        topCollabRecommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:topN]
        topCollabRecipeIds = [rec[0] for rec in topCollabRecommendations]
        collabRecommendations = recipesDf[recipesDf['recipe_id'].isin(topCollabRecipeIds)]
        collabRecommendations['source'] = 'Collaborative Filtering'
        collabRecommendations['predicted_rating'] = [rec[1] for rec in topCollabRecommendations]
        if not collabRecommendations.empty:
            return collabRecommendations[['name', 'predicted_rating']]
    else:
        userInteractions = interactionsDf[interactionsDf['user_id'] == userId]
        userProfile = np.zeros(len(recipesDf['tags_vector'].iloc[0]))
        for index, row in userInteractions.iterrows():
            recipeVector = np.array(recipesDf.loc[recipesDf['recipe_id'] == row['recipe_id'], 'tags_vector'].values[0])
            userProfile += row['rating'] * recipeVector
        userProfileTags = userProfile / len(userInteractions)
        recipeVectors = np.vstack(recipesDf['tags_vector'].values)
        similarityScores = cosine_similarity([userProfileTags], recipeVectors)[0]
        recipesDf['similarity'] = similarityScores
        recommendations = []
        userRatedRecipes = interactionsDf[interactionsDf['user_id'] == userId]['recipe_id'].unique()
        untried_recipes = recipesDf[~recipesDf['recipe_id'].isin(userRatedRecipes)].sort_values(by='similarity',                                                                                         ascending=False)
        for index, row in untried_recipes.head(5).iterrows():
            recipeVector = np.array(row['tags_vector'])
            matches = np.where((userProfileTags > 0) & (recipeVector > 0))[0]
            sortedMatches = sorted(matches, key=lambda i: userProfileTags[i] + recipeVector[i], reverse=True)
            matchingFeatures = [tagsMapping.get(str(i), f"Tag {i}") for i in sortedMatches[:3]]
            reason = f"Top similar tags: {', '.join(matchingFeatures)}" if matchingFeatures else "No clear match"
            recommendations.append({
                'recipe': row['name'],
                'similarity': row['similarity'],
                'reason': reason
            })
        return pd.DataFrame(recommendations)


with open('recipes_processed_key.json', 'r') as f:
    keyData = json.load(f)
tagsMapping = keyData['tags']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

userId = int(input('Enter user id: '))
topRecipes = recommendHybrid(userId, interactionsDf, recipesDf, tagsMapping)
print(topRecipes)
