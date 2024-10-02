from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd
import json
import ast
import numpy as np

recipesDf = pd.read_csv('recipes.csv')
interactionsDf = pd.read_csv('interactions.csv')


def convertStringToList(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


recipesDf['tags'] = recipesDf['tags'].apply(convertStringToList)
recipesDf['ingredients'] = recipesDf['ingredients'].apply(convertStringToList)


def collaborativeFiltering(userId, interactionsDf, recipesDf):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(interactionsDf[['user_id', 'recipe_id', 'rating']], reader)
    trainSet, testSet = train_test_split(data, test_size=0.2)
    model = SVD()
    model.fit(trainSet)
    ratedRecipes = interactionsDf[interactionsDf['user_id'] == userId]['recipe_id'].tolist()
    unratedRecipes = recipesDf[~recipesDf['recipe_id'].isin(ratedRecipes)]
    predictions = []
    for recipeId in unratedRecipes['recipe_id']:
        pred = model.predict(userId, recipeId)
        if pred.est > 4:
            predictions.append(recipeId)
    return predictions


def contentBasedFiltering(userId, interactionsDf, recipesDf, ingredients):
    data = interactionsDf[interactionsDf["user_id"] == userId]
    data = data.merge(recipesDf, left_on="recipe_id", right_on="recipe_id")
    personalValues = np.array([0] * len(ingredients))
    valuesIncremented = np.array([0] * len(ingredients))
    for i in range(len(data)):
        for ingredient in data.loc[i, "ingredients"]:
            valuesIncremented[ingredient] = valuesIncremented[ingredient] + 1
            added = (data.loc[i, "rating"] - personalValues[ingredient]) / valuesIncremented[ingredient]
            personalValues[ingredient] = personalValues[ingredient] + added
    return personalValues


def vectorizeRecipes(recipes, ingredients, collabRecommendations):
    recipesVectorized = []
    recommendedRecipes = recipes[recipes['recipe_id'].isin(collabRecommendations)]
    for i in range(len(recommendedRecipes)):
        recipeVector = np.array([0] * len(ingredients))
        for ingredient in recommendedRecipes.iloc[i]["ingredients"]:
            recipeVector[ingredient] = 1
        recipesVectorized.append((recommendedRecipes.iloc[i]["recipe_id"], recipeVector))
    return recipesVectorized


def generateRecommendations(personalVector, recipeVectors):
    ratings = []
    for recipe in recipeVectors:
        rating = np.dot(personalVector, recipe[1])
        ratings.append((recipe[0], rating))
    ratings.sort(key=lambda x: x[1], reverse=True)
    return ratings


def recommendHybrid(userId, interactionsDf, recipesDf, tagsMapping, ingredientsMapping, topN=5):
    collabRecommendations = collaborativeFiltering(userId, interactionsDf, recipesDf)
    personalVector = contentBasedFiltering(userId, interactionsDf, recipesDf, ingredientsMapping)
    recipeVectors = vectorizeRecipes(recipesDf, ingredientsMapping, collabRecommendations)
    personalRecommendations = generateRecommendations(personalVector, recipeVectors)
    combinedRecommendations = {}
    for rec in personalRecommendations:
        recipeId, score = rec
        combinedRecommendations[recipeId] = combinedRecommendations.get(recipeId, 0) + score
    sortedRecommendations = sorted(combinedRecommendations.items(), key=lambda x: x[1], reverse=True)
    topRecommendations = sortedRecommendations[:topN]
    recommendationsDf = pd.DataFrame(topRecommendations, columns=["recipe_id", "score"])
    recommendationsWithNames = recommendationsDf.merge(recipesDf[['recipe_id', 'name']], on="recipe_id", how="left")
    return recommendationsWithNames[['name', 'score']]


with open('recipes_processed_key.json', 'r') as f:
    keyData = json.load(f)
tagsMapping = keyData['tags']
ingredientsMapping = keyData['ingredients']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
userId = int(input('Enter user id: '))
ratedRecipes = interactionsDf[interactionsDf['user_id'] == userId]['recipe_id'].tolist()
ratedRecipesDf = pd.DataFrame(ratedRecipes, columns=["recipe_id"])
ratedRecipeNames = ratedRecipesDf.merge(recipesDf[['recipe_id', 'name']], on="recipe_id", how="left")
print(ratedRecipeNames)
topRecipes = recommendHybrid(userId, interactionsDf, recipesDf, tagsMapping, ingredientsMapping)
print(topRecipes)
