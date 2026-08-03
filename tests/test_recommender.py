import json

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


def make_recipes(ids, ingredients, tags):
    return pd.DataFrame({
        "id": ids,
        "ingredients": [json.dumps(i) for i in ingredients],
        "tags": [json.dumps(t) for t in tags],
    })


def make_interactions(rows):
    return pd.DataFrame(rows, columns=["user_id", "recipe_id", "rating"])


class TestParseReviews:
    def test_incremental_mean_matches_hand_computed_values(self, app_module):
        recipes = make_recipes(
            ids=[10, 11],
            ingredients=[[0, 1], [1, 2]],
            tags=[[0], [1]],
        )
        interactions = make_interactions([
            (1, 10, 4),
            (1, 11, 2),
        ])

        personalIngredients, personalTags = app_module.parseReviews(
            1, interactions, recipes, ingredients=[0, 1, 2], tags=[0, 1]
        )

        # ingredient 0: only rated once (via recipe 10) -> mean 4
        # ingredient 1: rated via recipe 10 (4) then recipe 11 (2) -> running mean 3
        # ingredient 2: only rated once (via recipe 11) -> mean 2
        assert personalIngredients == pytest.approx([4, 3, 2])
        # tag 0 rated once (4), tag 1 rated once (2)
        assert personalTags == pytest.approx([4, 2])

    def test_user_with_no_interactions_returns_zero_vectors_without_crashing(self, app_module):
        recipes = make_recipes(ids=[10], ingredients=[[0]], tags=[[0]])
        interactions = make_interactions([(1, 10, 5)])

        personalIngredients, personalTags = app_module.parseReviews(
            999, interactions, recipes, ingredients=[0], tags=[0]
        )

        assert personalIngredients == pytest.approx([0])
        assert personalTags == pytest.approx([0])


class TestGenerateRecommendations:
    """recipes[i] lines up positionally with column i of the vector matrices,
    matching how vectorizeRecipes/build_recipe_vectors.py produce them."""

    def _build(self):
        recipes = make_recipes(
            ids=[100, 101, 102, 103],
            ingredients=[[0, 1], [], [2], [0]],
            tags=[[0], [1], [], [0]],
        )
        # rows = ingredient/tag ids, cols = recipes, matching vectorizeRecipes' layout
        ingredientMatrix = csr_matrix(np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float32))
        tagMatrix = csr_matrix(np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
        ], dtype=np.float32))
        nIngredients = np.array([2, 0, 1, 1])
        nTags = np.array([1, 1, 0, 1])
        personalIngredients = np.array([5.0, 3.0, 1.0])
        personalTags = np.array([4.0, 2.0])
        return (
            recipes, ingredientMatrix, tagMatrix,
            nIngredients, nTags, personalIngredients, personalTags,
        )

    def test_zero_ingredients_or_tags_do_not_produce_nan_or_inf(self, app_module, monkeypatch):
        recipes, ingredientMatrix, tagMatrix, nIngredients, nTags, pIng, pTags = self._build()
        monkeypatch.setattr(app_module, "interactions", make_interactions([]))

        result = app_module.generateRecommendations(
            1, recipes, pIng, pTags, ingredientMatrix, tagMatrix, nIngredients, nTags
        )

        scores = [score for _id, score in result]
        assert not any(np.isnan(scores))
        assert not any(np.isinf(scores))
        # recipe 101 (zero ingredients) and 102 (zero tags) must still be scored, not dropped
        assert {id for id, _ in result} == {100, 101, 102, 103}

    def test_already_rated_recipes_are_excluded(self, app_module, monkeypatch):
        recipes, ingredientMatrix, tagMatrix, nIngredients, nTags, pIng, pTags = self._build()
        monkeypatch.setattr(app_module, "interactions", make_interactions([(1, 103, 5)]))

        result = app_module.generateRecommendations(
            1, recipes, pIng, pTags, ingredientMatrix, tagMatrix, nIngredients, nTags
        )

        assert 103 not in {id for id, _ in result}

    def test_results_are_ranked_highest_score_first(self, app_module, monkeypatch):
        recipes, ingredientMatrix, tagMatrix, nIngredients, nTags, pIng, pTags = self._build()
        monkeypatch.setattr(app_module, "interactions", make_interactions([]))

        result = app_module.generateRecommendations(
            1, recipes, pIng, pTags, ingredientMatrix, tagMatrix, nIngredients, nTags
        )

        scores = [score for _id, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_result_never_exceeds_twenty_five_recommendations(self, app_module, monkeypatch):
        n = 40
        recipes = make_recipes(
            ids=list(range(n)),
            ingredients=[[0] for _ in range(n)],
            tags=[[0] for _ in range(n)],
        )
        ingredientMatrix = csr_matrix(np.ones((1, n), dtype=np.float32))
        tagMatrix = csr_matrix(np.ones((1, n), dtype=np.float32))
        nIngredients = np.ones(n)
        nTags = np.ones(n)
        monkeypatch.setattr(app_module, "interactions", make_interactions([]))

        result = app_module.generateRecommendations(
            1, recipes, np.array([1.0]), np.array([1.0]),
            ingredientMatrix, tagMatrix, nIngredients, nTags
        )

        assert len(result) <= 25
