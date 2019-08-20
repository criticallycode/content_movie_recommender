### Content Based Recommendation Engine

This repo comes with a notebook, which explains the logic behind the various code blocks and where you can easily change the desired movie to see what kinds of recommendations you get.

This content based recommendation engine uses data from the TMDB movie dataset, which can be found [here](https://www.kaggle.com/tmdb/tmdb-movie-metadata/). The approach used in this recommender is the extraction of metadata keywords, which are then compiled into a keyword set and analyze for their similarity to the keyword set of other films. Various algorithms are used to calculate the similarity between keyword sets, and the votes of those metrics are returned, along with an averaged set of recommendations. 

This chunk handles the loading and transformation of the dataframes.

``` Python
# start off by loading in the data

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
meta = pd.read_csv('movies_metadata.csv')
links = pd.read_csv('links_small.csv')
ratings = pd.read_csv('ratings_small.csv')

def create_list(data):
    if isinstance(data, list):
        names = [i['name'] for i in data]
        return names
    else:
        return []

# dropping any nan values
meta['genres'] = meta['genres'].fillna('[]').apply(literal_eval).apply(create_list)

# get only non null values for the list of links
links = links[links['tmdbId'].notnull()]['tmdbId'].astype('int')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')

# there are some problematic entries in here, so let's drop them
# before casting as integers
meta = meta.drop([19730, 29503, 35587])
meta['id'] = meta['id'].astype('int')

# each of the dataframes has an 'id' columnm join them together to get complete data
meta = meta.merge(credits, on='id')
meta = meta.merge(keywords, on='id')
# get dataframe of only movies that have a tmdb id
new_meta = meta[meta['id'].isin(links)]
# let's see what the shape of our dataframe is
print(new_meta.shape)

```

This chunk handles the filtering/extraction of relevant features.

``` Python
# time to do content filtering: extract relevant features
# desired features - cast, crew keywords

new_meta['cast'] = new_meta['cast'].apply(literal_eval)
new_meta['crew'] = new_meta['crew'].apply(literal_eval)
new_meta['keywords'] = new_meta['keywords'].apply(literal_eval)

new_meta['cast_size'] = len(new_meta['cast'])
new_meta['crew_size'] = len(new_meta['crew'])

print(new_meta['cast_size'])
print(new_meta['crew_size'])

def get_director(data):
    for entry in data:
        if entry['job'] == 'Director':
            return entry['name']
    return np.nan

new_meta['director'] = new_meta['crew'].apply(get_director)

# need to get the other cast member names
new_meta['cast'] = new_meta['cast'].apply(create_list)
new_meta['cast'] = new_meta['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)

# now we need to create a list of keywords
new_meta['keywords'] = new_meta['keywords'].apply(create_list)

# make sure the director name is a string, replace and spaces
new_meta['cast'] = new_meta['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
new_meta['director'] = new_meta['director'].astype('str').apply(lambda x: str.lower(x.replace(" ","")))
# need to make three copies of the director to account for the weight of the other cast
new_meta['director'] = new_meta['director'].apply(lambda x: [x, x, x])

# make a series out of the keywords and get the total frequency/value counts for all the keywords
# stack function in pandas allows us to stack the keywords and count them
counts = new_meta.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
# name the elements in the series keyword
counts.name = 'keyword'
counts = counts.value_counts()
# let's check to see which words are the most common
print(counts[:5])
# drop any s' which only occur once
counts = counts[counts > 1]

# going to do some text pre-processing, stemming
stemmer = SnowballStemmer('english')

# create a function to filter the keywords
# if the input words are in the list of counts, add it to new list of words
def kw_filter(inp):
    words = []
    for i in inp:
        if i in counts:
            words.append(i)
    return words

# filter all the keywords
new_meta['keywords'] = new_meta['keywords'].apply(kw_filter)
# stem the keywords
new_meta['keywords'] = new_meta['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
# replace the spaces
new_meta['keywords'] = new_meta['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

# string everything together into a soup
new_meta['soup'] = new_meta['keywords'] + new_meta['cast'] + new_meta['director'] + new_meta['genres']
new_meta['soup'] = new_meta['soup'].apply(lambda x: ' '.join(x))

print(new_meta.columns.values)

nltk_stopwords = set(stopwords.words('english'))
count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words=nltk_stopwords)
# transform the data into a matrix
count_matrix = count_vect.fit_transform(new_meta['soup'])
```

The following portion is where the similarities are calculated using various similarity/distance algorithms.

``` Python
# get the cosine similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
euclidean = euclidean_distances(count_matrix, count_matrix)
euclidean = np.exp(-euclidean * 2)

sigmoid = sigmoid_kernel(count_matrix, count_matrix)

linear = linear_kernel(count_matrix, count_matrix)

rbf = rbf_kernel(count_matrix, count_matrix)
average = (cosine_sim + euclidean + sigmoid + linear + rbf)/5

# we need the indices of the movies in order to return recommendations
# reset the index as it may have changed thanks to creating new dataframes
new_meta = new_meta.reset_index()
# now we'll need a list of the titles with their corresponding indices
titles = new_meta['title']
# get the indices of the individual titles
indices = pd.Series(new_meta.index, index=new_meta['title'])
```

Here's the actual function that handles the return of recommendations.

``` Python
def get_recommendations(title, metric, number):
    # start by getting the index of the title
    idx = indices[title]
    # compute the sim scores of the title
    sim_scores = list(enumerate(metric[idx]))
    # we want to sort the scores in reverse order, top to bottom (most similar to least)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # return the top "X" results
    sim_scores = sim_scores[1:50]
    # get the first element - (the movie title)
    movie_indices = [i[0] for i in sim_scores]
    results = list(titles.iloc[movie_indices])[:number]
    return results
```

Finally, here's what calls that function and gets the recommendations for the desired movie.

``` Python
movie = 'Hot Fuzz'

cos_rec = get_recommendations(movie, cosine_sim, 20)
linear_rec = get_recommendations(movie, linear, 20)
euclidean_rec = get_recommendations(movie, euclidean, 20)
sigmoid_rec = get_recommendations(movie, sigmoid, 20)
rbf_rec = get_recommendations(movie, rbf, 20)
avg_rec = get_recommendations(movie, average, 20)

total_recommends = pd.DataFrame(columns=['Title'])
cos_recommends = pd.DataFrame(cos_rec, columns=["Title"])
linear_recommends = pd.DataFrame(linear_rec, columns=["Title"])
euc_recommends = pd.DataFrame(euclidean_rec, columns=["Title"])
sig_recommends = pd.DataFrame(sigmoid_rec, columns=["Title"])
rbf_recommends = pd.DataFrame(sigmoid_rec, columns=["Title"])

avg_recommends = pd.DataFrame(avg_rec, columns=['Title'])
recommends = [cos_recommends, linear_recommends, euc_recommends, sig_recommends, rbf_recommends]

for recommend in recommends:
    total_recommends = pd.concat([total_recommends, recommend], ignore_index=True)

total_recommends = total_recommends.groupby(total_recommends.columns.tolist()).size().reset_index().rename(columns={0:'Confidence'})
total_recommends = total_recommends.sort_values("Confidence", ascending=False)

print("This is our confidence weighted recommendations:")
print(total_recommends.head(20))

print()
print("This is our average recommendations: ")
print(avg_recommends.head(20))

```

This content based recommendation engine is inspired by the work of [Ibtesam Ahmed](https://www.kaggle.com/ibtesama)'s work on recommendation engines.

