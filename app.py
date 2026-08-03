import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, request

from recipe_vectors import load_raw_data

CACHE_PATH = "data/recipes_vectors.pkl"

interactions, recipes, ingredients, tags = load_raw_data()

with open(CACHE_PATH, "rb") as f:
    _cache = pickle.load(f)

if not np.array_equal(_cache["recipe_ids"], recipes["id"].to_numpy()):
    raise RuntimeError(
        f"{CACHE_PATH} is out of date with data/recipes_improved.csv (recipe ids "
        "don't match). Regenerate it by running build_recipe_vectors.py."
    )

recipes["url"] = _cache["url"]
recipesV = (_cache["ingredient_matrix"], _cache["tag_matrix"], _cache["nIngredients"], _cache["nTags"])

recipeNameByID = dict(zip(recipes["id"], recipes["name"]))
recipeUrlByID = dict(zip(recipes["id"], recipes["url"]))


def parseReviews(userID, interactions, recipes, ingredients, tags):
    data = interactions[interactions["user_id"] == userID]
    data = data.merge(recipes, left_on="recipe_id", right_on="id")
    personalIngredients = np.zeros(len(ingredients), dtype=np.float32)
    ingredientsIncremented = np.zeros(len(ingredients), dtype=np.uint16)
    personalTags = np.zeros(len(tags), dtype=np.float32)
    tagsIncremented = np.zeros(len(tags), dtype=np.uint16)

    parsedIngredients = data["ingredients"].apply(json.loads)
    parsedTags = data["tags"].apply(json.loads)

    for rowIngredients, rowTags, rating in zip(parsedIngredients, parsedTags, data["rating"]):
        for ingredient in rowIngredients:
            ingredientsIncremented[ingredient] = ingredientsIncremented[ingredient] + 1
            added = (rating - personalIngredients[ingredient]) / ingredientsIncremented[ingredient]
            personalIngredients[ingredient] = personalIngredients[ingredient] + added
        for tag in rowTags:
            tagsIncremented[tag] = tagsIncremented[tag] + 1
            added = (rating - personalTags[tag]) / tagsIncremented[tag]
            personalTags[tag] = personalTags[tag] + added

    return personalIngredients, personalTags


def generateRecommendations(userID, recipes, personalIngredients, personalTags, recipeIngredientVectors,
                            recipeTagVectors, nIngredients, nTags):
    iRatings = recipeIngredientVectors.T.dot(personalIngredients) / np.maximum(nIngredients, np.ones(len(nIngredients), dtype=np.int64))
    tRatings = recipeTagVectors.T.dot(personalTags) / np.maximum(nTags, np.ones(len(nTags), dtype=np.int64))
    ratings = (iRatings + tRatings) / 2
    recommend = np.argsort(ratings)[::-1][:25]
    alreadyRated = set(interactions.loc[interactions["user_id"] == userID, "recipe_id"])
    l = []
    for i in range(len(recommend)):
        id = recipes.loc[recommend[i], "id"]
        if id not in alreadyRated:
            l.append((id, ratings[recommend[i]]))
        if len(l) > 24:
            return l
    return l


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
        result.append(str(i + 1) + ': ' + recipeNameByID[id] + ',    score: ' + str(
            personalRecommendations[i][1]) +
                      ',    url: ' + recipeUrlByID[id])
    return jsonify({'Top recommendations': result})


def run_app():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))


if __name__ == "__main__":
    run_app()
