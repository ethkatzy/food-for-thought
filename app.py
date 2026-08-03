import html
import json
import os
import pickle
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flask import Flask, jsonify, render_template, request

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
recipesV = (
    _cache["ingredient_matrix"],
    _cache["tag_matrix"],
    _cache["nIngredients"],
    _cache["nTags"],
)

recipeNameByID = dict(zip(recipes["id"], recipes["name"]))
recipeUrlByID = dict(zip(recipes["id"], recipes["url"]))

MIN_USER_ID = int(interactions["user_id"].min())
MAX_USER_ID = int(interactions["user_id"].max())

# Neither a recipe's image nor its real food.com rating is in our dataset,
# so we scrape each recipe's food.com page on first request (og:image meta
# tag + the aggregateRating embedded in its JSON-LD block) and cache the
# result in memory (recipes get recommended to many users, so this pays off).
_RECIPE_META_CACHE = {}
_OG_IMAGE_RE = re.compile(r'name="og:image" content="([^"]+)"')
_AGGREGATE_RATING_RE = re.compile(
    r'"aggregateRating":\{"@type":"AggregateRating","ratingValue":"([^"]*)","reviewCount":"([^"]*)"\}'
)
_IMAGE_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FoodForThoughtBot/1.0)"}
_IMAGE_FETCH_TIMEOUT = 5
_PAGE_FETCH_MAX_BYTES = 2_000_000
# Recipes with no photo of their own get food.com's generic share-card
# graphic back from og:image instead — treat that as "no image".
_GENERIC_IMAGE_MARKER = "fdc-shareGraphic"
# Some users have rated thousands of recipes; cap what's sent to the
# frontend's "recipes rated by this user" table (highest-rated first, since
# those are what most influenced the recommendations).
_RATED_RECIPES_LIMIT = 200


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


def generateRecommendations(userID, recipes, personalIngredients, personalTags,
                            recipeIngredientVectors, recipeTagVectors, nIngredients, nTags):
    iRatings = recipeIngredientVectors.T.dot(personalIngredients) / np.maximum(
        nIngredients, np.ones(len(nIngredients), dtype=np.int64)
    )
    tRatings = recipeTagVectors.T.dot(personalTags) / np.maximum(
        nTags, np.ones(len(nTags), dtype=np.int64)
    )
    ratings = (iRatings + tRatings) / 2
    recommend = np.argsort(ratings)[::-1][:25]
    alreadyRated = set(interactions.loc[interactions["user_id"] == userID, "recipe_id"])
    recs = []
    for i in range(len(recommend)):
        id = recipes.loc[recommend[i], "id"]
        if id not in alreadyRated:
            recs.append((id, ratings[recommend[i]]))
        if len(recs) > 24:
            return recs
    return recs


def getRatedRecipes(userID):
    userRatings = interactions[interactions["user_id"] == userID]
    rated = [
        {
            "id": int(recipeID),
            "name": recipeNameByID.get(recipeID, "Unknown recipe"),
            "rating": int(rating),
            "url": recipeUrlByID.get(recipeID),
        }
        for recipeID, rating in zip(userRatings["recipe_id"], userRatings["rating"])
    ]
    rated.sort(key=lambda r: (-r["rating"], r["name"]))
    total = len(rated)
    return rated[:_RATED_RECIPES_LIMIT], total


def fetchRecipeMeta(recipeID, url):
    if recipeID in _RECIPE_META_CACHE:
        return _RECIPE_META_CACHE[recipeID]

    meta = {"image": None, "foodComRating": None, "foodComReviewCount": None}
    try:
        req = urllib.request.Request(url, headers=_IMAGE_FETCH_HEADERS)
        with urllib.request.urlopen(req, timeout=_IMAGE_FETCH_TIMEOUT) as resp:
            page = resp.read(_PAGE_FETCH_MAX_BYTES).decode("utf-8", errors="ignore")

        imageMatch = _OG_IMAGE_RE.search(page)
        if imageMatch and _GENERIC_IMAGE_MARKER not in imageMatch.group(1):
            meta["image"] = html.unescape(imageMatch.group(1))

        ratingMatch = _AGGREGATE_RATING_RE.search(page)
        if ratingMatch:
            try:
                reviewCount = int(ratingMatch.group(2))
                if reviewCount > 0:
                    meta["foodComRating"] = float(ratingMatch.group(1))
                    meta["foodComReviewCount"] = reviewCount
            except ValueError:
                pass
    except (urllib.error.URLError, OSError, TimeoutError):
        pass

    _RECIPE_META_CACHE[recipeID] = meta
    return meta


def fetchRecipeMetas(recommendations):
    uncached = [id for id, _score in recommendations if id not in _RECIPE_META_CACHE]
    if uncached:
        with ThreadPoolExecutor(max_workers=min(10, len(uncached))) as executor:
            list(executor.map(lambda id: fetchRecipeMeta(id, recipeUrlByID[id]), uncached))
    return {id: _RECIPE_META_CACHE.get(id, {}) for id, _score in recommendations}


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', min_user_id=MIN_USER_ID, max_user_id=MAX_USER_ID)


@app.route('/process', methods=['POST'])
def process():
    user = request.form.get('user_id', '').strip()
    try:
        userID = int(user)
    except ValueError:
        return jsonify({
            'error': f'User ID must be an integer between {MIN_USER_ID} and {MAX_USER_ID}.'
        }), 400
    if not (MIN_USER_ID <= userID <= MAX_USER_ID):
        return jsonify({
            'error': f'User ID must be between {MIN_USER_ID} and {MAX_USER_ID}.'
        }), 400

    personalV = parseReviews(userID, interactions, recipes, ingredients, tags)
    personalRecommendations = generateRecommendations(
        userID, recipes, personalV[0], personalV[1],
        recipesV[0], recipesV[1], recipesV[2], recipesV[3],
    )
    top = personalRecommendations[:10]
    metas = fetchRecipeMetas(top)
    result = [
        {
            'id': int(id),
            'name': recipeNameByID[id],
            'score': round(float(score), 2),
            'url': recipeUrlByID[id],
            'image': metas.get(id, {}).get('image'),
            'foodComRating': metas.get(id, {}).get('foodComRating'),
            'foodComReviewCount': metas.get(id, {}).get('foodComReviewCount'),
        }
        for id, score in top
    ]
    ratedRecipes, ratedRecipesTotal = getRatedRecipes(userID)
    return jsonify({
        'recommendations': result,
        'ratedRecipes': ratedRecipes,
        'ratedRecipesTotal': ratedRecipesTotal,
    })


def run_app():
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)))


if __name__ == "__main__":
    run_app()
