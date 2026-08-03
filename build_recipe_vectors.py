import pickle

from recipe_vectors import build_urls, load_raw_data, vectorizeRecipes

CACHE_PATH = "data/recipes_vectors.pkl"


def main():
    interactions, recipes, ingredients, tags = load_raw_data()
    ingredient_matrix, tag_matrix, nIngredients, nTags = vectorizeRecipes(recipes, ingredients, tags)
    url = build_urls(recipes)

    cache = {
        "recipe_ids": recipes["id"].to_numpy(),
        "ingredient_matrix": ingredient_matrix,
        "tag_matrix": tag_matrix,
        "nIngredients": nIngredients,
        "nTags": nTags,
        "url": url.to_numpy(),
    }
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"Wrote {CACHE_PATH} ({len(recipes)} recipes)")


if __name__ == "__main__":
    main()
