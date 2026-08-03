import json

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def load_raw_data():
    interactions = pd.read_csv("data/interactions_processed.csv", header=0)
    recipes = pd.read_csv("data/recipes_improved.csv", header=0)
    tags = pd.read_json("data/recipes_processed_key.json")["tags"]
    ingredients = pd.read_json("data/recipes_processed_key.json")["ingredients"]
    return interactions, recipes, ingredients, tags


def vectorizeRecipes(recipes, ingredients, tags):
    parsedIngredients = recipes["ingredients"].apply(json.loads)
    parsedTags = recipes["tags"].apply(json.loads)

    nIngredients = np.zeros(len(recipes), dtype=np.uint8)
    nTags = np.zeros(len(recipes), dtype=np.uint8)

    ingredientRows, ingredientCols = [], []
    for i, rawIng in enumerate(parsedIngredients):
        ing = set(rawIng)
        ingredientRows.extend(ing)
        ingredientCols.extend([i] * len(ing))
        nIngredients[i] = len(rawIng)

    tagRows, tagCols = [], []
    for i, rawTags in enumerate(parsedTags):
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


def build_urls(recipes):
    nameSlug = recipes["name"].astype(str).str.split().str.join("-")
    return (
        "https://www.food.com/recipe/"
        + nameSlug
        + np.where(nameSlug != "", "-", "")
        + recipes["id"].astype(str)
    )
