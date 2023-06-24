from .data_utils import data_utils
from .column_types import get,Number,FuzzyString,Categorical,String,DateTime
from .clauses import Clause
from .conditionmaps import conditions
from .get_time_period_dates import get_time_period_dates
from datetime import date

from transformers import TFBertForQuestionAnswering, BertTokenizer,DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf

from rake_nltk import Rake
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
lem=lemmatizer.lemmatize
#qa_model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',padding=True)
qa_model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")

#from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
#import tensorflow as tf

#tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
#model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

#question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

#inputs = tokenizer(question, text, return_tensors="tf")
#outputs = model(**inputs)

#answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
#answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

#predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
#tokenizer.decode(predict_answer_tokens)


def extract_keywords_from_doc(doc, phrases=True, return_scores=False):
    if phrases:
        r = Rake()
        if isinstance(doc, (list, tuple)):
            r.extract_keywords_from_sentences(doc)
        else:
            r.extract_keywords_from_text(doc)
        if return_scores:
            return [(b, a) for a, b in r.get_ranked_phrases_with_scores()]
        else:
            return r.get_ranked_phrases()
    else:
        if not isinstance(doc, (list, tuple)):
            doc = [doc]
        ret = []
        for x in doc:
            for t in nltk.word_tokenize(x):
                if t.lower() not in stop_words:
                    ret.append(t)
        return ret

    

def extract_keywords_from_query(query, phrases=True):
    if not phrases:
        tokens = nltk.pos_tag(nltk.word_tokenize(query))
        return [t[0] for t in tokens if  t[0].lower() not in stop_words and t[1] != '.']
    kws = extract_keywords_from_doc(query, phrases=True)
    tags = dict(nltk.pos_tag(nltk.word_tokenize(query)))
    filtered_kws = []
    for kw in kws:
        kw_tokens = nltk.word_tokenize(kw)
        for t in kw_tokens:
            if t in tags and tags[t][0] in ('N', 'C', 'R', 'S'):
                filtered_kws.append(kw)
                break
    return filtered_kws
    


def qa(docs, query, return_score=False, return_all=False, return_source=False, sort=False):
    if isinstance(docs, (list, tuple)):
        answers_and_scores = [qa(doc, query, return_score=True) for doc in docs]
        if sort:
            sort_ids = list(range(len(docs)))
            sort_ids.sort(key=lambda i: -answers_and_scores[i][1])
            answers_and_scores = [answers_and_scores[i] for i in sort_ids]
        if return_source and sort:
            docs = [docs[i] for i in sort_ids]
        if not return_score:
            answers = [a[0] for a in answers_and_scores]
        else:
            answers = answers_and_scores
        if return_source:
            if return_score:
                answers = [answers[i] + (docs[i],) for i in range(len(docs))]
            else:
                answers = [(answers[i], docs[i]) for i in range(len(docs))]
        return answers if return_all else answers[0]

    doc = docs
    
    
    
    #question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    #inputs = qa_tokenizer(query, doc, return_tensors="tf")
    #outputs = qa_model(**inputs)

    #print("OUTPUTS ",outputs)
    #answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    #answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    #predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    #print("ANS TOKENS ",predict_answer_tokens)
    
    #answer= qa_tokenizer.decode(predict_answer_tokens)
    #print("ANSWER ",answer)
    
    if True:
        print("QA ",query,doc)
        encoding = qa_tokenizer.encode_plus(query, doc, padding=True, truncation=True, return_tensors="tf")
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        outputs = qa_model(input_ids, attention_mask=attention_mask)
        #print("OUTPUTS ",outputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        
        start_probs = tf.nn.softmax(start_scores, axis=1).numpy()[0]
        end_probs = tf.nn.softmax(end_scores, axis=1).numpy()[0]

        
        start_index = tf.argmax(start_probs).numpy()
        end_index = tf.argmax(end_probs).numpy()

        #start_index = tf.argmax(start_scores, axis=1).numpy()[0]
        #end_index = tf.argmax(end_scores, axis=1).numpy()[0]

        #start_score = start_scores[0, start_index].numpy().item()
        #end_score = end_scores[0, end_index].numpy().item()
        
        #start_score = tf.reduce_max(start_scores, axis=1).numpy()[0]
        #end_score = tf.reduce_max(end_scores, axis=1).numpy()[0]

        
        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]
        answer=qa_tokenizer.decode(predict_answer_tokens)
        print("ANSWER ",answer)
        print("ANSWER IDX ",answer_start_index,answer_end_index)
        answer = answer.replace(' #', '').replace('#', '').replace('[CLS]', '').replace('[SEP]', '').replace(' - ','-')
        input_kws = set(extract_keywords_from_query(query.lower(), phrases=False))
        answer_kws = set(extract_keywords_from_query(answer.lower(), phrases=False))
        num_input_kws = len(input_kws)
        input_kws.update(answer_kws)
        if len(input_kws) == num_input_kws:
            score = 0
        else:
            score = start_probs[start_index] * end_probs[end_index]
        
        
        #answer_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids.numpy()[0, start_index:end_index+1])
        #answer = qa_tokenizer.convert_tokens_to_string(answer_tokens)

        print("Answer:", answer)
        #print("Start Score:", start_probs)
        #print("End Score:", end_probs)
        print("Total Score:", start_probs[start_index] * end_probs[end_index])
        
        
        return answer,score
    
    if False:
        input_ids = qa_tokenizer.encode(query, doc)
        print("INPUT IDS ",input_ids, len(input_ids) ) 
        if len(input_ids) > 512:
            sentences = nltk.sent_tokenize(doc)
            if len(sentences) == 1:
                if return_score:
                    return '', -1000
                else:
                    return ''
            else:
                return qa(sentences, query, return_score=return_score)
        sep_index = input_ids.index(qa_tokenizer.sep_token_id)
        num_seg_a = sep_index + 1
        num_seg_b = len(input_ids) - num_seg_a
        segment_ids = [0]*num_seg_a + [1]*num_seg_b
        #print("SEG IDS ",segment_ids, len(segment_ids) ) 

        #print("INP ",tf.constant([input_ids]))
        #print("SEG ",tf.constant([segment_ids]))

        #start_scores, end_scores = qa_model(tf.constant([input_ids]),token_type_ids=tf.constant([segment_ids]))
        start_scores, end_scores = qa_model(tf.constant([input_ids]),token_type_ids=tf.constant([segment_ids])).values()
        #print("START SCORES  ",start_scores)
        #print("END SCORES  ",end_scores)

        tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
        print("TOKENS ",tokens)
        num_input_tokens = sep_index + 1
        print("NUM INP TOKENS ",num_input_tokens)

        answer_start = tf.argmax(start_scores[0][num_input_tokens:]) + num_input_tokens
        answer_end = tf.argmax(end_scores[0][num_input_tokens:]) + num_input_tokens

        answer = ' '.join(tokens[int(answer_start.numpy()): int(answer_end.numpy() + 1)])
        answer = answer.replace(' #', '').replace('#', '').replace('[CLS]', '').replace('[SEP]', '')
        if not return_score:
            return answer
        
        print("PREDICTED Answer:", answer)
        
        input_kws = set(extract_keywords_from_query(query.lower(), phrases=False))
        answer = answer.replace(' #', '').replace('#', '').replace('[CLS]', '').replace('[SEP]', '')
        answer_kws = set(extract_keywords_from_query(answer.lower(), phrases=False))
        num_input_kws = len(input_kws)
        input_kws.update(answer_kws)
        
        #print("Answer:", answer)
        #print("Start Score:", start_scores)
        #print("End Score:", end_scores)
        #print("Start Score:",start_scores[0][answer_start])
        #print("End Score:", end_scores[0][answer_end])
        
        if len(input_kws) == num_input_kws:
            score = 0
        else:
            score = float((start_scores[0][answer_start] + end_scores[0][answer_end]))
        
        print("Score ",score)
        return answer, score
            
    #return '', -1000        
    #return answer, score

    
stop_words = stopwords.words('english')
stop_words.append('whose')


def _norm(x):
    x = x.strip()
    while '  ' in x:
        x = x.replace('  ', ' ')
    return x.lower()


def _underscore(x):
    return _norm(x).replace(' ', '_')


def _find(lst, sublst):
    print("FIND ",lst)
    print("FIND ",sublst)
    for i in range(len(lst)):
        flag = False
        for j in range(len(sublst)):
            if sublst[j] != lst[i + j]:
                flag = True
                break
        if not flag:
            return i
    return -1

def _window_overlap(s1, e1, s2, e2):
    return s2 <= e1 if s1 <s2 else s1 <= e2


class Nlp:
    def __init__(self,data_dir,schema_dir):
        self.data_dir = data_dir
        self.schema_dir = schema_dir
        self.values= {}
        #if isinstance(self.schema_dir,dict):
        #    self.schema=schema_dir
        
        self.data_process=data_utils(data_dir, schema_dir)
        
        #self.valuesfile =self.data_process.valuesfile
        #self.data_process.create_values()
        #with open(self.valuesfile, 'r') as f:
        #    self.values = json.load(f)

    def slot_fill(self, df, q):
        # example: slot_fill(get_csvs()[2], "how many emarati men of age 22 died from stomach cancer in 2012")
        self.schema = self.data_process.get_schema_for_csv(df)
        schema=self.schema
        print(schema)
        print("QUESTION ",q)
        def _is_numeric(typ):
            # TODO
            return issubclass(get(typ), Number)

        slots = []
        mappings = {}
        for col in schema['columns']:
            colname = col['name']
            if 'keywords' in col.keys():
                keyword=col['keywords'][0]
                q=q.replace(colname,keyword)
            else:
                keyword=colname
            if colname == 'index':
                continue
            coltype = col['type']
            if coltype == "Categorical":
                mappings[colname] = col["mapping"]
    
            print("COLNAME ",colname)
            print("COLTYPE ",coltype)
            if _is_numeric(coltype):
                colquery="number of {}?".format(keyword)
            else:
                colquery="which {}?".format(keyword)
            
            print("COL QUERY", colquery)
            
            val, score = qa(q, colquery, return_score=True)
            #print(val,score)
            vt =  nltk.word_tokenize(val)
            start_idx = _find(nltk.word_tokenize(q), vt)
            end_idx = start_idx + len(vt) - 1
            print("filling slots %s %s %s",colname, val, score)
            slots.append((colname, coltype, val, score, start_idx, end_idx))
            #print("SLOTS2 ",slots)
        
        #print("SLOTS BEFORE SORT ",slots)
        slots.sort(key=lambda x: -x[3])
        print("SLOTS AFTER SORT ",slots)
        print("SLOTS AFTER SORT ",slots[0][-3])
        windows = []
        slots_filtered = []
        for s in slots:
            #if s[-2] < 0:
            if s[-2] < 0 or s[-3] < 0.5:
                continue
            win = s[-2:]
            flag = False
            for win2 in windows:
                if _window_overlap(*(win + win2)):
                    flag = True
                    break
            if flag:
                continue
            windows.append(win)
            slots_filtered.append(s[:-2])
            
        slots = slots_filtered
        print("SLOTS ",slots)
        #print(mappings)
        ret = []
        for s in slots:
            if s[1] == "FuzzyString":
                #vals = self.values[s[0]]
                #fs = FuzzyString(vals, exclude=s[0].split('_'))
                #val = fs.adapt(s[2])
                print("FUZZY SLOT ",s)
                val=s[2]
            elif s[1] == "String":    
                val=s[2]
            elif s[1] == "Timestamp":    
                val=s[2]    
            elif s[1] == "Categorical":
                print("CAT SLOT ",s)
                cat = Categorical(mappings[s[0]])
                val = cat.adapt(s[2])
            elif _is_numeric(s[1]):
    
                val = get(s[1])().adapt(s[2], context=q, allowed_kws=[s[0]])
            else:
                val = get(s[1])().adapt(s[2])
            if val is not None:
                ret.append((s[0], s[1], val, s[3]))
        
        return ret
    
    
    
    def cond_map(self,s):
        #map the conditional operators for <,> etc from respective words like greater than,less than,etc
        conds=conditions
        condflag=False
        words=[i for i in s.split() if not i.isdigit()]
        nums=[i for i in s.split() if i.isdigit()]
        for word in words:
            for k,v in conds.items():
                  if word in v:
                      if len(nums)==1:
                          num=nums[0]
                          s=f'{k} {num}'
                          condflag=True
                      else:
                          if "BETWEEN {} AND {}" in k:
                              k=k.format(nums[0],nums[1])
                              s=f'{k}'
                              condflag=True
        return s,condflag
        
    def kword_extractor(self,q):
        ret=[]
        for t in nltk.word_tokenize(q):
            if t.lower() not in stop_words:
                kwd=lem(t.lower())
                ret.append(kwd)
        return ret               
    
    def unknown_slot_extractor(self,schema,sf_columns,ex_kwd):
        #extracts the key if exists from a query, whose value is not mapped with the csv
        maxcount=0
        unknown_slots={"slots":[],"main_slot":None}
        flag=False
        
        for col in schema["columns"]:
            if col["name"] not in sf_columns:
                col_kwds=[]    
                if '_' in col["name"]:
                    col_kwds.extend(col["name"].split("_"))
                else:
                    col_kwds.append(col["name"])
                if 'keywords' in col.keys():
                    col_kwds.extend(col["keywords"])
                    
                col_kwds=[lem(i.lower()) for i in col_kwds]
            
                count=len(set(col_kwds) & set(ex_kwd))
                if count>0:
                    unknown_slots["slots"].append(col["name"])
                if count>maxcount:
                    maxcount=count
                    unknown_slots["main_slot"]=col["name"]
                   
                    flag=True  if col["type"] in ["Date","Integer","Decimal","Age"] else False
                    
        return unknown_slots,flag
    

    
    
    def time_period_filter(self,schema_col,time_period):
        
        filter=''
        if time_period.count("-") == 5:
            parts = time_period.split("-", 3)
            part1 = '-'.join(parts[:3]).strip()
            part2 = '-'.join(parts[3:]).strip()
            filter="{} BETWEEN {} AND {}".format(schema_col['name'],part1,part2)
        else:
            filter="{} = {}".format(schema_col['name'],time_period)
        
        return filter
        
        
    def get_sql_query(self, q):
        #df = self.data_dir
        #get sql query by adding each clauses back to back by aggregate type classification and  entity extraction from slot_fill
        #csvData = "data/Cancer Death - Data.csv"
        csvData = "data/oura_sleep.xxx"
        
        inputLabels=[
          {
            "entity_group": "SCONJ",
            "score": 0.5785049200057983,
            "word": "how",
            "start": 0,
            "end": 3
          },
          {
            "entity_group": "PRON",
            "score": 0.9942967295646667,
            "word": "i",
            "start": 8,
            "end": 9
          },
          {
            "entity_group": "VERB",
            "score": 0.988246500492096,
            "word": "sleep",
            "start": 10,
            "end": 15
          },
#          {
#            "entity_group": "TIME",
#            "score": 0.9617880582809448,
#            "word": "last night",
#            "start": 16,
#            "end": 26
#          },
            {
    "entity_group": "DATE",
    "score": 0.9812600612640381,
    "word": "last week",
    "start": 16,
    "end": 25
  }
        ]
        #how did i sleep last night?
        #012345678901234567890123456
        #0         1         2
        
        
        current_date = date.today()
        possible_timeperiods= get_time_period_dates(current_date)
        print(possible_timeperiods)
        #for term, iso_date in result.items():
        #    print(f'{term}: {iso_date}')
        timePeriod=""
        for item in inputLabels:
            if item["entity_group"] == "TIME" or item["entity_group"] == "DATE":
                #timePeriod=possible_timeperiods[item["word"]]
                timePeriod=item["word"]
                q=q.replace(timePeriod,"")
                #item["word"] = "replacement word"

                #original_string = "Hello, World!"
                #new_string = original_string.replace("World", "Python")

                #print(new_string)
        #print(array)

        sf = self.slot_fill(csvData, q)
        
        print("GOT SLOTS ",sf)
        schema=self.schema  
        sf_columns=[i[0] for i in sf]
        ex_kwd=self.kword_extractor(q)
        unknown_slots,flag=self.unknown_slot_extractor(schema,sf_columns,ex_kwd)
        
        
        clause=Clause()
        question=""
        distinct = False
        question = clause.adapt([q], distinct=distinct)
        print("QUESTION ",question)
        if flag:
            for col in schema["columns"]:
                if "summable" in col.keys() and col["name"] in unknown_slots["main_slot"]:
                    question=clause.adapt([q],inttype=True,summable=True)

                    break
        if question not in "SELECT {} FROM {}":
            unknown_slots=unknown_slots['main_slot']
        else:
           unknown_slots= ','.join(unknown_slots['slots'])
        if not(unknown_slots and unknown_slots.strip()):
            unknown_slots='*'
        question=question.format(unknown_slots,schema["name"].lower())
        print("QUESTION2 ",question)
        print("SCHEMA 0 ",schema['columns'][0])
        
        sql=question
        sub_clause=''' WHERE {} = '{}' '''
        for i,s in enumerate(sf):
            condflag=False
            col,val=s[0],s[2]
            typ = get(s[1])
            if i>0:
                sub_clause='''AND {} = '{}' '''
            if issubclass(typ,Number):
                val,condflag=self.cond_map(val)
            subq=sub_clause.format(col, val)
            if condflag:
                subq=subq.replace('=','')
                subq=subq.replace("'","")
            
            question+=subq   #repeatedly concatenates the incoming entities in sql syntax         
    
    
        #message = "My name is {} and I'm {} years old.".format(name, age)
        #print(message)
        print("QUESTION3 ",question)

        if sql != question:
            sql=question.strip()+" AND "+self.time_period_filter(schema['columns'][0],possible_timeperiods[timePeriod])
        else:
            sql+=" WHERE "+self.time_period_filter(schema['columns'][0],possible_timeperiods[timePeriod])
            
        return sql