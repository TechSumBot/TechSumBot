# -*- coding: utf-8 -*-
from pandas.io import clipboards
import pymysql 
import pandas as pd 
import pickle
import signal
import time
# -- start mysql use: mysql --local-infile=1 -u root -p
# -- password: 123456
def sql_connection():
   db = pymysql.connect(host='localhost',
                        user='root',
                        password='123456',
                        database='SO')

   cursor = db.cursor(pymysql.cursors.DictCursor)
   return cursor

def set_timeout(num, callback):
  def wrap(func):
    def handle(signum, frame): 
      raise RuntimeError
    def to_do(*args, **kwargs):
      try:
        signal.signal(signal.SIGALRM, handle) 
        signal.alarm(num) 
      #   print('start alarm signal.')
        r = func(*args, **kwargs)
      #   print('close alarm signal.')
        signal.alarm(0) 
        return r
      except RuntimeError as e:
        callback()
    return to_do
  return wrap

def after_timeout(): 
  print("Time out!")

@set_timeout(3, after_timeout) 
def get_query_name(Id):
   cursor = sql_connection()
   sql = '''
   select Title from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_parientid(Id):
   cursor = sql_connection()
   sql = '''
   select ParentId from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()
   if results:
      return results
   else: 
      return 'Null'

@set_timeout(3, after_timeout) 
def get_votes(Id):
   cursor = sql_connection()
   sql = '''
   select Score from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_query_body(Id):
   cursor = sql_connection()
   sql = '''
   select Body from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results



@set_timeout(3, after_timeout) 
def get_answer_body(Id):
   cursor = sql_connection()
   sql = '''
   select Body from posts
   where Id= '''+Id+''' 
   and PostTypeId  = 2'''
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_answer_votes(Id):
   cursor = sql_connection()
   sql = '''
   select Title from posts
   where Id='''+Id
   cursor.execute(sql)
   results = cursor.fetchall()

   return results

def get_related_answer():
   cursor = sql_connection()

   sql = '''
   SELECT a.RelatedPostId,(sum(b.AnswerCount)+(SELECT distinct posts.AnswerCount FROM post_links,posts WHERE post_links.RelatedPostId=posts.Id and a.RelatedPostId=posts.Id))TotalAnswerCount 
   FROM post_links as a, posts as b
   WHERE LinkTypeId = 3 
   and a.PostId=b.Id 
   -- and Posts.AnswerCount>=1
   and b.Tags like '%python%'
   group by a.RelatedPostId order by sum(b.AnswerCount) desc 
   '''
   cursor.execute(sql)
   results = cursor.fetchall()

   data = pd.DataFrame(list(results),columns=['Original post ID','number of related answers'])
   data.to_csv('data.csv',index=False)
   print(data)

def get_relatedID(id,pl):
   cursor = sql_connection()
   sql = '''
   select b.PostId
   from posts as a, post_links as b
   where b.RelatedPostId='''+str(id)+'''
   and b.PostId=a.Id
   and a.Tags like '%'''+str(pl)+'''%'
   and b.LinkTypeId = 3
   group by b.PostId
   '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results


def get_related_posts(pl):
   cursor = sql_connection()
   sql = '''select b.RelatedPostId,
   (sum(a.AnswerCount)+(select a.AnswerCount from posts as a where a.Id=b.RelatedPostId))TotalAnswerCount
   from posts as a, post_links as b 
   where b.PostId=a.Id 
   and a.Tags like '%<'''+str(pl)+'''>%' 
   and b.LinkTypeId = 3  
   group by b.RelatedPostId 
   order by (sum(a.AnswerCount)+(select a.AnswerCount from posts as a where a.Id=b.RelatedPostId))  desc;'''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results


def get_data(id):
   cursor = sql_connection()
   sql = '''select * from posts
   where ParentId ='''+str(id)
   cursor.execute(sql)
   result = cursor.fetchall()
   return result

def get_id(file):
   java_candidate = file[(file['TotalAnswerCount']>=10) & (file['TotalAnswerCount']<=18)]
   # candidate 
   # [RelatedPostId, TotalAnswerCount]
   return java_candidate

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_duplicate_pairs(pl):
   cursor = sql_connection()
   sql = '''
   select  a.RelatedPostId,a.postid  from PostLinks as a, Posts as b  
   where a.LinkTypeId = 3
   and a.PostId=b.Id 
   and b.Tags like '%<'''+pl+'''>%'
   '''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

def get_tag_posts(tag):
   cursor = sql_connection()
   sql = '''
   select id, AcceptedAnswerId from posts where posttypeid=1 
   and tags like '%<'''+str(tag)+'''>%' 
   and AcceptedAnswerId is not null 
   '''
   print(sql)
   cursor.execute(sql)
   results = cursor.fetchall()
   return results

@set_timeout(3, after_timeout) 
def get_neg_answer(parentid):
   cursor = sql_connection()
   sql = '''
   select body, id,score from posts where ParentId = '''+parentid+''' 
   and PostTypeId  = 2 
   and Score<0
   '''
   cursor.execute(sql)
   results = cursor.fetchall()
   return results