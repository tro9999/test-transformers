import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords, wordnet 
from nltk.stem import PorterStemmer

#nltk.download('wordnet')
#nltk.download('stopwords')

ps = PorterStemmer().stem
syns = wordnet.synsets
stop_words = list(set(stopwords.words('english')))


class data_utils:
    def __init__(self,data_dir,schema_dir):
        self.data_dir = data_dir
        self.schema_dir = schema_dir
        
    def rename(self,name):
        renamed=""
        for i in name.lower():
            if i.isalnum():
                renamed+=i
            else:
                if not renamed:
                    renamed='_'
                if not renamed[-1]=="_":
                    renamed+="_"
        return renamed
    
    def get_dataframe(self,csv_path):
        data= pd.read_csv(csv_path)
        columns=data.columns.tolist()
        columns=[self.rename(i) for i in columns]
        data.columns=columns
        return data
        """
        {'name': 'cancer_death',
 'columns': [{'name': 'nationality',
   'mapping': {'National': ['national',
     'nationals',
     'citizen',
     'citizens',
     'emarati',
     'emaratis'],
    'Expatriate': ['expatriate',
     'foreigner',
     'foreigners',
     'immigrant',
     'immgrants',
     'foreign']},
   'type': 'Categorical'},
  {'name': 'gender',
   'mapping': {'Male': ['male', 'males', 'man', 'men', 'boys'],
    'Female': ['female', 'females', 'woman', 'women', 'girls']},
   'type': 'Categorical'},
  {'name': 'cancer_site',
   'keywords': ['type of cancer', 'cancer location'],
   'type': 'FuzzyString'},
  {'name': 'death_count',
   'keywords': ['died', 'death', 'dead'],
   'summable': 'True',
   'type': 'Integer'},
  {'name': 'index', 'type': 'Integer', 'keywords': ['index']},
  {'name': 'year', 'type': 'Date', 'keywords': ['year']},
  {'name': 'age', 'type': 'Age', 'keywords': ['age']}],
 'keywords': ['cancer', 'death', 'end']}
        """
        
    def get_schema_for_csv(self,df_path):
        data=self.get_dataframe(df_path)
        print("DATA ",data)
        dfname=os.path.splitext(os.path.basename(df_path))[0]
        columns=data.columns.tolist() 
        columns=[self.rename(i) for i in columns]
        if "unnamed" in columns[0].lower():
            columns[0]="index"
        data.columns=columns   
        
        #print(data,dfname)
        # if isinstance(self.schema_dir,dict):
        #        schema=self.schema_dir
        with open(os.path.join(self.schema_dir, df_path[len(self.data_dir) + 1:-4]) + '.json', 'r') as f:
            schema=json.load(f)
            
        schema_keywords=[]
        schema["name"]=self.rename(schema["name"])
        if "columns" not in schema.keys():
            schema["columns"]=[]
        else:
            for col in schema["columns"]:
                col["name"]=self.rename(col["name"])
        if "keywords" not in schema.keys():
            for name in schema["name"].split("_"):
                schema_syns=syns(ps(name))
                schema_keywords.extend(list(set(i.lemmas()[0].name().lower().replace("_"," ") for i in  schema_syns)))

        if schema_keywords:
            schema_keywords=[i for i in schema_keywords if i not in stop_words]
            schema["keywords"]=schema_keywords
        
        #print(schema)
        types=data.dtypes.apply(lambda x:self.rename(x.name)).to_dict()
        for k,v in types.items():
            if 'int' in v:
                types[k]="Integer"
            if 'float' in v:
                types[k]="Decimal"
            if k.lower()=="age":
                    types[k]="Age"
            if  k.lower() in ("year","month","week"):
                    types[k]="Date"
            if 'object' in v:
                for col in schema["columns"]:
                    if "mapping" in col:
                        types[col["name"]]="Categorical"
                    else:
                        types[k]="FuzzyString"

            collist=[]          
            for col in schema["columns"]:
                col["name"]=self.rename(col["name"])
                collist.append(col["name"])
                col["type"]=types[col["name"]]

            for column in columns:
                column=self.rename(column)
                if column not in collist:
                    schema["columns"].append({"name":column,"type":types[column],"keywords":[" ".join(column.lower().split('_'))]})
        return schema
        #print(schema)