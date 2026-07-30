# food-for-thought

## What it does (rough notes, cleaning up later)

It's a recipe recommender. You give it a food.com user id, it looks at that
user's past recipe ratings and returns a personalized top-25 (well, top-10
shown) list of recipes they haven't rated yet, ranked by predicted score.

## How it works (`app.py`)

Data loaded at startup:
- `data/interactions_processed.csv` — user_id, recipe_id, rating (past ratings)
- `data/recipes_improved.csv` — recipe id, name, ingredient ids, tag ids
- `data/recipes_processed_key.json` — lookup tables mapping ingredient/tag ids to names

Pipeline:
1. **`parseReviews(userID, ...)`** — pulls all of that user's rated recipes,
   then for every ingredient and every tag that shows up in those recipes,
   computes a running average of the ratings the user gave recipes containing
   it. Ends up with two vectors: "how much this user tends to like each
   ingredient" and "... each tag", based only on their own history.
2. **`vectorizeRecipes(...)`** — runs once at startup over all ~232k recipes.
   Builds two sparse binary matrices (ingredients x recipes, tags x recipes)
   marking which ingredients/tags are in which recipe, plus per-recipe counts
   of how many ingredients/tags each has (used for normalizing later).
3. **`generateRecommendations(userID, ...)`** — dot-products the user's
   preference vectors against every recipe's ingredient/tag matrix, divides
   by ingredient/tag count per recipe (so long ingredient lists don't just
   win by volume), averages the ingredient-score and tag-score, sorts all
   recipes by that, and returns the top 25 the user hasn't already rated.
4. `create_url` just builds a food.com-style URL from the recipe name + id
   for display purposes.

Flask app:
- `/` — plain HTML form, type in a user id.
- `/process` (POST) — runs the pipeline above for that user id, returns JSON
  with the top 10 recommendations (name, score, url).
- Runs on port 5002, started in a background thread at import time (not
  behind `if __name__ == "__main__"`).

Core idea: no cosine similarity/matrix factorization/ML model — it's a
content-based approach using a user's own historical average rating per
ingredient/tag as the "preference weight," applied to every recipe's
ingredient/tag composition.
