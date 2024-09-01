import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial.distance import correlation
import csv
import datetime
import logging
import requests
import time
import os.path
import sys
import pandas as pd
import numpy as np
import re, math
from collections import Counter
from csv import reader
from csv import DictReader

WORD = re.compile(r'\w+')

#applying cosine similarity for finding similarities between user interests and places
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])
     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)
     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
    
#converting (category)text to vector
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

#remove spaces from the category column of dataset
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
        
#finding similarity between users
def similarity(user1,user2):
    try:
        user1=np.array(user1)-np.nanmean(user1)
        user2=np.array(user2)-np.nanmean(user2)
        commonItemIds=[i for i in range(len(user1)) if user1[i]>0 and user2[i]>0]
        if len(commonItemIds)==0:
           return 0
        else:
           user1=np.array([user1[i] for i in commonItemIds])
           user2=np.array([user2[i] for i in commonItemIds])
           return correlation(user1,user2)
    except ZeroDivisionError:
        print("You can't divide by zero!")
        
#Finding nearest neighbours
def nearestNeighbourRatings(activeUser,K):
    try:
        similarityMatrix=pd.DataFrame(index=userItemRatingMatrix.index,columns=['Similarity'])
        
        for i in userItemRatingMatrix.index:
            #similarity function called to build similarity matirx
            similarityMatrix.loc[i]=similarity(userItemRatingMatrix.loc[activeUser],userItemRatingMatrix.loc[i])
        
        similarityMatrix=pd.DataFrame.sort_values(similarityMatrix,['Similarity'],ascending=[0])
        nearestNeighbours=similarityMatrix[:K]
        neighbourItemRatings=userItemRatingMatrix.loc[nearestNeighbours.index]
        predictItemRating=pd.DataFrame(index=userItemRatingMatrix.columns, columns=['Rating'])
        for i in userItemRatingMatrix.columns:
            predictedRating=np.nanmean(userItemRatingMatrix.loc[activeUser])
            for j in neighbourItemRatings.index:
                if userItemRatingMatrix.loc[j,i]>0:
                   predictedRating += (userItemRatingMatrix.loc[j,i]-np.nanmean(userItemRatingMatrix.loc[j]))*nearestNeighbours.loc[j,'Similarity']
                predictItemRating.loc[i,'Rating']=predictedRating
    except ZeroDivisionError:
        print("You can't divide by zero!")            
    return predictItemRating

#FUNCTION THAT RETURNS LIST OF TOP RECOMMENDATIONS
def topNRecommendations(activeUser,N):
    try:
        predictItemRating=nearestNeighbourRatings(activeUser,10)
        placeAlreadyWatched=list(userItemRatingMatrix.loc[activeUser].loc[userItemRatingMatrix.loc[activeUser]>0].index)
        predictItemRating=predictItemRating.drop(placeAlreadyWatched)
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:N]
        topRecommendationTitles=(placeInfo.loc[placeInfo.itemId.isin(topRecommendations.index)])
    except ZeroDivisionError:
        print("You can't divide by zero!")
    a1=len(list(topRecommendationTitles.title))
    l1=int(0.5*a1)
    lt1=list(topRecommendationTitles.title)
    q = 0
    lt11=lt1[q:l1]
    #print(a1)
    #print(l1)
    return lt11

def load_data():
     data = []
     for x in range(len(final_dest)):
        data.append({
                "origin": src,
                "destination": final_dest[x],
                "mode": "driving",
                "traffic_model": "best_guess",
                "departure_time": "now"
            })
     return data

#function to make request to get distance matrix
def make_request(base_url, api_key, origin, destination, mode, traffic_model, departure_time):
    url = "{base_url}/maps/api/distancematrix/json" \
          "?key={api_key}" \
          "&origins={origin}" \
          "&destinations={destination}" \
          "&mode={mode}" \
          "&traffic_model={traffic_model}" \
          "&departure_time={departure_time}".format(base_url=base_url,
                                                    api_key=api_key,
                                                    origin=origin,
                                                    destination=destination,
                                                    mode=mode,
                                                    traffic_model=traffic_model,
                                                    departure_time=departure_time)
    result = requests.get(url)
    return result.json()

#main function
def main():
    data = load_data()
    n=0
    with open(RESULT_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['request_time', 'origin', 'destination', 'origin addresses', 'destination addresses','traffic_model', 'Departure time','Distance (meters)','Distance (text)','Duration (text)','Duration in traffic (text)'])
        
        for t in data:
            time.sleep(0)
            request_time = datetime.datetime.now()
            dm_res = make_request(BASE_URL, API_KEY, t['origin'], t['destination'], t['mode'], t['traffic_model'], t['departure_time'])
       
            if dm_res['status'] == 'REQUEST_DENIED':
                if dm_res['error_message'] == 'The provided API key is invalid or token limit exceeded.':
                    print(dm_res['error_message'])
                    break
            n+=1
            try:
                dm_distance = dm_res['rows'][0]['elements'][0]['distance']
                dm_duration = dm_res['rows'][0]['elements'][0]['duration']
                dm_duration_in_traffic = dm_res['rows'][0]['elements'][0]['duration_in_traffic']
                origin_addresses = dm_res['origin_addresses']
                destination_addresses = dm_res['destination_addresses']

            except Exception as exc:
                print("%s) Please check if the address or coordinates in this line are correct" % n)
                # logging.error(str(exc))
                continue

            csvwriter.writerow([request_time,t['origin'],t['destination'],origin_addresses,destination_addresses,
                                t['traffic_model'],t['departure_time'],dm_distance['value'],dm_distance['text'],
                                dm_duration['text'],dm_duration_in_traffic['text']])
    sortingdist()
    
#function that sorts destinations with distance
def sortingdist():
            df = pd.read_csv("result.csv")
            sorted_df = df.sort_values(by=["Distance (meters)"], ascending=True)
            sorted_df.to_csv('result.csv', index=False)
            #print("%s ==>%s : [Distance: %s | DURATION: %s]" % (csvD['origin'],csvD['destination'],csvD['Distance (text)'],csvD['Duration in traffic (text)']))
            with open('result.csv', 'r') as read_obj:
                # pass the file object to DictReader() to get the DictReader object
                csv_dict_reader = DictReader(read_obj)
                # iterate over each line as a ordered dictionary
                n=1
                for csvData in csv_dict_reader:
                  print("%s ==>%s : [Distance: %s | DURATION: %s]" % (csvData['origin'],csvData['destination'],csvData['Distance (text)'],csvData['Duration in traffic (text)']))
                  n=n+1
                  if(n==6):
                    break
                
#user preference
print("ENTER ANY CATEGORY FROM THESE:\n1.wildlife\n2.heritage\n3.pilgirmage\n4.park\n5.museum\n6.snow\n7.adventure\n8.nature")
text1 = input("\nENTER YOUR INTEREST OF LOCATIONS: ")  
vector1 = text_to_vector(text1)

metadata = pd.read_csv('data.csv', low_memory=False)
C = metadata['p_rating'].mean()
m = metadata['count'].quantile(0.75)

#Calulating weighted rating of places
def weighted_rating(x, m=m, C=C):
    v = x['count']
    R = x['p_rating']
    # Calculation based on the Bayesian Rating Formula
    return (v/(v+m) * R) + (m/(m+v) * C)

metadata['category'] = metadata['category'].apply(clean_data)
#storing weighted rating 
metadata['score'] = metadata.apply(weighted_rating,axis=1)
cos=[]

for i in list(metadata['category']):
    text2 = i
    vector2 = text_to_vector(text2)
    cosine = get_cosine(vector1, vector2)
    cos.append(cosine)
    
metadata['cosine']=cos
x=metadata['cosine']>0.0
rec=pd.DataFrame(metadata[x])
rec=rec.sort_values('score',ascending=False)
a2=len(list(rec['title']))
l2=int(0.5*a2)
lt2=list(rec['title'])
p = 0
ltt2=lt2[p:l2]
cont_dest=ltt2


data=pd.read_csv('data1.csv')
placeInfo=pd.read_csv('data.csv')
data=pd.merge(data,placeInfo,left_on='itemId',right_on="itemId")
userIds=data.userId
userIds2=data[['userId']]
data.loc[0:10,['userId']]
data=pd.DataFrame.sort_values(data,['userId','itemId'],ascending=[0,1])
userItemRatingMatrix=pd.pivot_table(data, values='rating',index=['userId'], columns=['itemId'])

#GETTING USER ID
activeUser=int(input("\nENTER USERID: "))

coll_dest=topNRecommendations(activeUser,10)


#FINAL TOTAL DESTINATIONS OBTAINED
final_dest=list(set(cont_dest+coll_dest))


src=input("ENTER YOUR CURRENT LOCATION: ")
print("\nTOP 5 DESTINATIONS WITH DISTANCE FROM YOUR CURRENT LOCATION IS DISPLAYED")
    
if __name__ == '__main__':
    BASE_URL = "https://api.distancematrix.ai"
    RESULT_FILE_PATH = "result.csv"
    API_KEY =#API KEY
    main()



