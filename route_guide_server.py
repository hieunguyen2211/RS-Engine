# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the gRPC route guide server."""
from cgitb import text
import asyncio
from sqlalchemy import create_engine
import logging
import numpy as np
from sqlalchemy import text
import pandas as pd
import grpc
from content_base import ContentBase
import rs_pb2
import rs_pb2_grpc
import urllib
import os
from dotenv import load_dotenv
load_dotenv()

class RecommendationServicer(rs_pb2_grpc.RecommendationServicer):
    """Provides methods that implement functionality of route guide server."""
    def __init__(self):
        self.server = os.getenv('SERVER')
        self.user = os.getenv('USER_DB')
        self.password = os.getenv('PASSWORD')
        self.port = os.getenv('PORT')
        self.database = os.getenv('DATABASE_NAME')
        print('server', self.server)
        print('server', self.user)
        print('server', self.password)
        print('server', self.port)
        print('server', self.database)

        yhat, users, data = InitDb(self)
        self.yhat = yhat
        self.users = users
        self.data = data

    def TrackChange(self, request, context):
        print('track change');
        try:
          yhat, users, data = InitDb(self)
          self.yhat = yhat
          self.users = users
          self.data = data
          return rs_pb2.Check(message='success')
        except:
          return rs_pb2.Check(message='failed')

    def GetItemRecommended(self, request, context):
        indexUserId = self.get_Index_user(request.id)
        itemIdsRated = self.yhat[:, indexUserId]
        output = np.asarray([idx for idx, element in enumerate(itemIdsRated) if (element > 1.5)])
        print('output 1', output)

        if output.size == 0:
            # Get Most popular
            itemIds = self.GetMostPularItem()
            return rs_pb2.ItemResponse(itemIds=itemIds)
        else:
            # get rated item
            itemIds = self.data[output, 0]
            print('itemIdsRated', itemIdsRated)
            print('output', output)
            #return index of a sorted list
            indexItemSortedIds = sorted(range(len(itemIds)), key=lambda k: itemIds[k], reverse=True)
            print('item sorted', indexItemSortedIds)
            result = self.data[:,0][indexItemSortedIds]
            print(result)
            
            
            return rs_pb2.ItemResponse(itemIds=map(str, result)) #return sorted List ids of item by uuid

    def GetMostPularItem(self):
        sumArr = np.asarray(list(map((lambda  x: sum(x)), self.yhat)))
        # return index of a sorted list
        indexItemSortedIds = sorted(range(len(sumArr)), key=lambda k : sumArr[k], reverse=True)
        return self.data[:,0][indexItemSortedIds] #return sorted List ids of item by uuid

    def get_Index_user(self, userId):
        ids = np.where(self.users == int(userId))[0][0]
        return ids
def mapData(item, l_tags):
      i_map = list(map((lambda x:  0 if x['name'] not in item[1] else 1), l_tags))
      i_map.insert(0, item[0])
      return np.asarray(i_map)

def InitDb(self):
    # params = urllib.parse.quote_plus("DRIVER=ODBC+Driver+17+for+SQL+Server;"
    #                                  "SERVER="+self.server+";"
    #                                  "DATABASE="+self.database+";"
    #                                  "UID="+self.user+";"
    #                                  "PWD="+self.password)
    
    engine = create_engine("postgresql://"+self.user+":"+self.password+"@"+self.server+"/"+self.database+"")
  
    with engine.connect() as connection:
        result = connection.execute(text('SELECT services.id, service_categories."enName" as name FROM services inner join service_service_categories on services.id = service_service_categories."serviceId" inner join service_categories on service_categories.id = service_service_categories."serviceCategoryId"'))
        test = [{column: value for column, value in rowproxy.items()} for rowproxy in result] #Return List of Dict
        res = {}
        for item in test:
          res.setdefault(item['id'], []).append(item['name'])
    with engine.connect() as connection:
      tag = connection.execute(text('SELECT id, "enName" as name from service_categories'));
      l_tag = [{column: value for column, value in rowproxy.items()} for rowproxy in tag]
      data = list(res.items())
      a = np.asarray(list(map((lambda x: mapData(x, l_tag)), data)))
      numberOfItem = len(l_tag)
      X_train_counts = a[:, -(numberOfItem - 1):]

    with engine.connect() as connection:
      users = connection.execute(text("SELECT id from customers"))
      users = pd.DataFrame(users)

    # tfidf
    tfidf = ContentBase.getTfidf(X_train_counts)
    print(tfidf)
    rate_train = getUserRatingMatrix(engine)
    print("rate_train", rate_train)

    d = tfidf.shape[1]  # data dimension
    n_users = users.shape[0]
    W = np.zeros((d, n_users))
    b = np.zeros((1, n_users))
    W, b = ContentBase.GetRidgeRegression(self=ContentBase, n_users=np.asarray(users), rate_train=rate_train,
                                          tfidf=tfidf, W=W, b=b, index_arr=a[:, 0])
    Yhat = tfidf.dot(W) + b
    # rmse = ContentBase.RMSE(ContentBase, np.asarray(users), rate_train, Yhat)
    # print(rmse)
    return Yhat, users, a
def getUserRatingMatrix(engine):
    with engine.connect() as connection:
        result = connection.execute(text('select "customerId", "serviceId", rating from customer_ratings'))
        test = [{column: value for column, value in rowproxy.items()} for rowproxy in result]
        df = pd.DataFrame(test)
        return df.values

async def serve() -> None:
  server = grpc.aio.server()

  rs_pb2_grpc.add_RecommendationServicer_to_server(
    RecommendationServicer(), server)

  listen_addr = '[::]:50051'
  server.add_insecure_port(listen_addr)
  logging.info("Starting server on %s", listen_addr)
  await server.start()
  await server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
