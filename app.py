from ast import literal_eval

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from scipy.sparse import coo_matrix

interactions = pd.read_csv("data/interactions_processed.csv", header=0)
recipes = pd.read_csv("data/recipes_improved.csv", header=0)
tags = pd.read_json("data/recipes_processed_key.json")["tags"]
ingredients = pd.read_json("data/recipes_processed_key.json")["ingredients"]


def parseReviews(userID, interactions, recipes, ingredients, tags):
    data = interactions[interactions["user_id"] == userID]
    data = data.merge(recipes, left_on="recipe_id", right_on="id")
    personalIngredients = np.array([0] * len(ingredients), dtype=np.float32)
    ingredientsIncremented = np.array([0] * len(ingredients), dtype=np.uint16)
    personalTags = np.array([0] * len(tags), dtype=np.float32)
    tagsIncremented = np.array([0] * len(tags), dtype=np.uint16)

    for i in range(len(data)):
        for ingredient in literal_eval(data.loc[i, "ingredients"]):
            ingredientsIncremented[ingredient] = ingredientsIncremented[ingredient] + 1
            added = (data.loc[i, "rating"] - personalIngredients[ingredient]) / ingredientsIncremented[ingredient]
            personalIngredients[ingredient] = personalIngredients[ingredient] + added
        for tag in literal_eval(data.loc[i, "tags"]):
            tagsIncremented[tag] = tagsIncremented[tag] + 1
            added = (data.loc[i, "rating"] - personalTags[tag]) / tagsIncremented[tag]
            personalTags[tag] = personalTags[tag] + added

    return personalIngredients, personalTags


def vectorizeRecipes(recipes, ingredients, tags):
    nIngredients = np.array([0] * len(recipes), dtype=np.uint8)
    nTags = np.array([0] * len(recipes), dtype=np.uint8)

    ingredientRows, ingredientCols = [], []
    for i in range(len(recipes)):
        rawIng = literal_eval(recipes.loc[i, "ingredients"])
        ing = set(rawIng)
        ingredientRows.extend(ing)
        ingredientCols.extend([i] * len(ing))
        nIngredients[i] = len(rawIng)

    tagRows, tagCols = [], []
    for i in range(len(recipes)):
        rawTags = literal_eval(recipes.loc[i, "tags"])
        ts = set(rawTags)
        tagRows.extend(ts)
        tagCols.extend([i] * len(ts))
        nTags[i] = len(rawTags)

    recipesIngredientsVectorized = coo_matrix(
        (np.ones(len(ingredientRows), dtype=np.float32), (ingredientRows, ingredientCols)),
        shape=(len(ingredients), len(recipes)),
    ).tocsr()
    recipesTagsVectorized = coo_matrix(
        (np.ones(len(tagRows), dtype=np.float32), (tagRows, tagCols)),
        shape=(len(tags), len(recipes)),
    ).tocsr()

    return recipesIngredientsVectorized, recipesTagsVectorized, nIngredients, nTags


def generateRecommendations(userID, recipes, personalIngredients, personalTags, recipeIngredientVectors,
                            recipeTagVectors, nIngredients, nTags):
    iRatings = recipeIngredientVectors.T.dot(personalIngredients) / np.maximum(nIngredients, np.array([1] * len(nIngredients)))
    tRatings = recipeTagVectors.T.dot(personalTags) / np.maximum(nTags, np.array([1] * len(nTags)))
    ratings = (iRatings + tRatings) / 2
    recommend = np.argsort(ratings)[::-1][:25]
    l = []
    for i in range(len(recommend)):
        id = recipes.loc[recommend[i], "id"]
        if id not in interactions[interactions["user_id"] == userID].loc[:, "recipe_id"].values:
            l.append((id, ratings[recommend[i]]))
        if len(l) > 24:
            return l
    return l


recipesV = vectorizeRecipes(recipes, ingredients, tags)


def create_url(name, id):
    url = 'https://www.food.com/recipe/'
    for word in str(name).split():
        url += word + '-'
    url += str(id)
    return url


recipes['url'] = recipes.apply(lambda x: create_url(x['name'], x['id']), axis=1)

app = Flask(__name__)


@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>Enter User ID</h1>
            <form id="userForm" method="POST" action="/process">
                <label for="user_id">User ID:</label>
                <input type="text" id="user_id" name="user_id" required>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    '''


@app.route('/process', methods=['POST'])
def process():
    user = request.form['user_id']
    personalV = parseReviews(int(user), interactions, recipes, ingredients, tags)
    personalRecommendations = generateRecommendations(int(user), recipes, personalV[0], personalV[1], recipesV[0],
                                                      recipesV[1], recipesV[2], recipesV[3])
    result = []
    for i in range(10):
        id = personalRecommendations[i][0]
        result.append(str(i + 1) + ': ' + recipes.loc[recipes['id'] == id, 'name'].values[0] + ',    score: ' + str(
            personalRecommendations[i][1]) +
                      ',    url: ' + recipes.loc[recipes['id'] == id, 'url'].values[0])
    return jsonify({'Top recommendations': result})


def run_app():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))


if __name__ == "__main__":
    run_app()
