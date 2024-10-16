import numpy as np
import pandas as pd

interactions= pd.read_csv("interactions_processed.csv", header= 0)
recipes= pd.read_csv("recipes_improved.csv", header= 0)
tags= pd.read_json("recipes_processed_key.json")["tags"]
ingredients= pd.read_json("recipes_processed_key.json")["ingredients"]


#functions

from ast import literal_eval #parses .csv string "lists" to actual python lists

def parseReviews(userID, interactions, recipes, ingredients, tags): #parses a user's submitted into vectors of average rating for each ingredient and tag
    #print(recipes.dtypes)
    data= interactions[interactions["user_id"]== userID]
    data= data.merge(recipes, left_on= "recipe_id", right_on= "id")
    print(data["name"].head(15))
    personalIngredients= np.array([0]*len(ingredients), dtype= np.float32)
    ingredientsIncremented= np.array([0]*len(ingredients), dtype=np.uint16)
    personalTags= np.array([0]*len(tags), dtype=np.float32)
    tagsIncremented= np.array([0]*len(tags), dtype=np.uint16)

    for i in range(len(data)):
        for ingredient in literal_eval(data.loc[i,"ingredients"]): #process ingredients
            #print(ingredient)
            ingredientsIncremented[ingredient]= ingredientsIncremented[ingredient]+1
            added= (data.loc[i,"rating"]-personalIngredients[ingredient])/ingredientsIncremented[ingredient]
            personalIngredients[ingredient]= personalIngredients[ingredient]+added

        #for i in range(len(data)):
        for tag in literal_eval(data.loc[i,"tags"]): #process tags
            tagsIncremented[tag]= tagsIncremented[tag]+1
            added= (data.loc[i,"rating"]-personalTags[tag])/tagsIncremented[tag]
            personalTags[tag]= personalTags[tag]+added

    return personalIngredients, personalTags # outputs 2 vectors of length n = |ingredients| and length t = |tags| respectively.



def vectorizeRecipes(recipes, ingredients, tags): # parses the recipes data into binary matrices of ingredients and tags in each recipe
    nIngredients=np.array([0]*len(recipes), dtype= np.uint8)
    nTags=np.array([0]*len(recipes), dtype= np.uint8)
    recipesIngredientsVectorized= np.zeros((len(ingredients),len(recipes)),np.float32)
    recipesTagsVectorized= np.zeros((len(tags),len(recipes)),np.float32)


    for i in range(len(recipes)):
        #process ingredients
        count=0
        ing=literal_eval(recipes.loc[i,"ingredients"])
        for j in range(len(ing)):
            recipesIngredientsVectorized[ing[j],i]=1
            count=count+1
        nIngredients[i]= count

    for i in range(len(recipes)):
        #process tags
        count2=0
        ts=literal_eval(recipes.loc[i,"tags"])
        for j in range(len(ts)):
            recipesTagsVectorized[ts[j],i]=1
            count2=count2+1
        nTags[i]= count2

    return recipesIngredientsVectorized, recipesTagsVectorized, nIngredients, nTags #  outputs two matrices of with dimensions mxn and mxt, and two length m = |recipes| vectors.


def generateRecommendations(userID, recipes, personalIngredients, personalTags, recipeIngredientVectors, recipeTagVectors, nIngredients, nTags): #generates the personal recommendations for a user
    iRatings= np.dot(recipeIngredientVectors.T, personalIngredients)/ nIngredients
    tRatings= np.dot(recipeTagVectors.T, personalTags) / np.maximum( nTags, np.array([1]*len(nTags)))
    #print("calc done")

    ratings=(iRatings+tRatings)/2
    recommend= np.argsort(ratings)[::-1][:25]
    l=[]
    #print(interactions[interactions["user_id"]== userID].loc[:,"recipe_id"].values)
    for i in range(len(recommend)):
        #print(recipes.loc[recommend[i], "id"])
        id= recipes.loc[recommend[i], "id"]
        if id not in interactions[interactions["user_id"]== userID].loc[:,"recipe_id"].values:
            l.append((id, ratings[recommend[i]]))
        if len(l)>24:
            return l
    return l

recipesV= vectorizeRecipes(recipes, ingredients, tags)

from flask import Flask, render_template, request, jsonify
from threading import Thread

def create_url(name,id):
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
    personalV= parseReviews(int(user),interactions, recipes, ingredients,tags)
    personalRecommendations= generateRecommendations(int(user), recipes, personalV[0],personalV[1], recipesV[0],recipesV[1],recipesV[2], recipesV[3])
    result = []
    for i in range(10):
        id = personalRecommendations[i][0]
        result.append(str(i+1) + ': ' + recipes.loc[recipes['id'] == id, 'name'].values[0] + ',    score: ' + str(personalRecommendations[i][1]) +
                      ',    url: ' + recipes.loc[recipes['id'] == id, 'url'].values[0])
    return jsonify({'Top recommendations': result})

# Run Flask in a separate thread
def run_app():
    app.run(host='0.0.0.0', port=5002)

# Start Flask server in a separate thread
thread = Thread(target=run_app)
thread.start()
