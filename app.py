from flask import Flask, request, redirect, url_for, Response, abort, render_template, send_from_directory
from flask_mysqldb import MySQL
from flask_cors import CORS
from werkzeug.utils import secure_filename
import vprnn.models
import vprnn.utils
import torch
import torch.nn
import torch.nn.functional as F
import pdb

from collections import namedtuple
import socket
import json
import glob
import os
import pkg_resources
import re
import random
import uuid
from math import log10
from scipy.stats import entropy

from gevent.pywsgi import WSGIServer
from gevent import monkey
from gevent import signal

from cs_rnn_choose import *

import logging
from logging.handlers import RotatingFileHandler

class Predictor(object):    
    def __init__(self, dictionary, device=None):
        self.dictionary = dictionary
        self.device = device
    
    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        """
        string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub("\'s", " \'s", string)
        string = re.sub("\'m", " \'m", string)
        string = re.sub("\'ve", " \'ve", string)
        string = re.sub("n\'t", " n\'t", string)
        string = re.sub("\'re", " \'re", string)
        string = re.sub("\'d", " \'d", string)
        string = re.sub("\'ll", " \'ll", string)
        string = re.sub(",", " , ", string)
        string = re.sub("!", " ! ", string)
        string = re.sub("\(", " ( ", string)
        string = re.sub("\)", " ) ", string)
        string = re.sub("\?", " ? ", string)
        string = re.sub("\s{2,}", " ", string)
        return string.strip().lower().split(" ")

    def package(self, query):
        tokens = self.clean_str(query)
        dat = []
        for t in tokens:
            if t in self.dictionary.word2idx:
                dat.append(self.dictionary.word2idx[t])
            else:
                dat.append(self.dictionary.word2idx['<pad>'])    
        dat = dat[:500]
        with torch.set_grad_enabled(False):
            dat = torch.tensor(dat, dtype=torch.long).unsqueeze(0)
        return dat.t()
    
    def predict(self, query, model):
        model.eval()
        query = self.package(query)
        if self.device is not None:
            query = query.to(self.device)
        with torch.no_grad():
            hidden = model.init_hidden(1)
            output, _ = model.forward(query, hidden)
            confidence = output.view(1, -1)
            logits = F.log_softmax(confidence, dim=1)
        return confidence, logits
    
monkey.patch_all()

DEVICE = None
WS_VERSION = "1.3.0"
                                           
with open('config.json') as conf_json:
    conf = json.load(conf_json)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
rfh = RotatingFileHandler(conf['log_file'],
                          maxBytes=100000,
                          backupCount=10,
                          encoding="UTF-8")
logger.addHandler(rfh)

cs_host = conf['cs_host'] 
cs_port = conf['cs_port'] 
cs_bufsize = conf['cs_bufsize'] 

app = Flask(__name__)
app.config['MYSQL_USER'] = conf['db_user']
app.config['MYSQL_PASSWORD'] = conf['db_pass']
app.config['MYSQL_DB'] = conf['db_db']
app.config['MYSQL_HOST'] = conf['db_host']
db = MySQL(app)
app.config['MAX_CONTENT_LENGTH'] = 9999999
CORS(app)


logger.info("Building vocabularies...")
dictionary = vprnn.utils.Dictionary(path=conf['dict_path'])
predictor = Predictor(dictionary, device=DEVICE)
label_text_path = conf["label_path"]
lbl_txt = open(label_text_path).readlines()
idx_to_lbl = []
lbl_to_idx = {}
for x in lbl_txt:
    lbl, text = int(x.split(' ')[0]), ' '.join(x.split(' ')[1:]).strip()
    idx_to_lbl.insert(lbl, text)
    lbl_to_idx[text] = lbl

cs_rnn_map = {}
label_mapping = open(conf['label_map']).readlines()
for line in label_mapping:
    cs_label, nn_label = line.split('\t')[0].strip(), line.split('\t')[1].strip()
    cs_rnn_map[cs_label] = nn_label

logger.info("Done.")

logger.info("Loading RNN model...")
logger.info("Vocab size: %d", len(dictionary))
model_conf = {'ntoken':len(dictionary),
                'dictionary':dictionary,
                'ninp':300,
                'word-vector':None,
                'nhid':300,
                'nlayers':1,
                'attention-hops':2,
                'attention-unit':350,
                'dropout':0.5,
                'nfc':512,
                'nclasses':359,
                'pooling':'all'
               }
rnn_model = vprnn.models.RnnAtt(model_conf)
rnn_model.load_state_dict(torch.load(conf['rnn_model_path'], map_location='cpu'))
logger.info("Done.")


decider = CS_RNN_chooser(conf['cs_rnn_classifier'], conf['cs_rnn_vectorizer'])

def rnn_predict(query):
    predicted_query = ""
    class_conf, class_prob = predictor.predict(query, rnn_model)
    return class_prob, class_conf

def best_idx(tensor):
    # postprocess output
    _, argmax = torch.max(tensor, 1)
    idx = int(argmax.data[0])
    # print(label_map[int(argmax.data[0])])
    return idx

def query(class_num):
    return idx_to_lbl[class_num]

def class_num(query):
    return lbl_to_idx[query]

def process_vars(line):
    mvars = {"chiefcomplaint":"pain",
             "complaint2":"frequent urination",
             "patientname":"wilkins",
             "patientlastname":"wilkins",
             "currentmedication1":"ibuprofen",
             "currentjob":"mechanic",
             "patientfirstname":"jim"}
    if "$" not in line:
        return line
    else:
        assignments = re.findall(r'\$[a-zA-Z0-9]+\?',line)
        for assignment in assignments:
            varname = assignment[1:-1]
            #placeholder = assignment.split("=")[1]
            ##for each occurrence of placeholder in line, replace with varname lookup
            try:
                varvalue = mvars[varname]
            except KeyError:
                varvalue = varname
            line = re.sub(" "+re.escape(assignment)+" "," "+varvalue+" ", line)
            # line = line.split("$")[0] #cut at first variable assignment
        return line
                                                                                            
def process_match(why):
    logger.debug(why)
    # match = extract_interp_re.search(why)
    left, right = why.split(":", 1)
    template_name = right.split("(",1)[0].strip()
    #if match:
    if template_name != "":
        # template_name = match.group(1).strip()
        # processed = process_vars(raw)
        return template_name
    else:
        return "!!did not extract template!!"
    

## TODO: exception on broken connections?
def cs_exchange(usr_first, usr_last, patient_num, msg):
    message = usr_first+"_"+usr_last+":patient"+str(patient_num)+"\0\0"+msg+"\0"
    cs_sock = socket.socket()
    cs_sock.connect((cs_host, cs_port))
    sent = 0
    while sent < len(message):
        sent += cs_sock.send(bytearray(message[sent:], 'UTF-8'))
    chunks = []
    reply = cs_sock.recv(cs_bufsize)
    chunks.append(reply)
    while reply != b'':
        reply = cs_sock.recv(cs_bufsize)
        chunks.append(reply)
    cs_sock.close()
    return b''.join(chunks).decode("utf-8")

@app.route("/config/", methods=['GET'])
def show_client_config():
    if 'client' in request.args and 'setup' in request.args:
        client_type = request.args.get('client','')
        setup = request.args.get('setup','')
        response_dict = {}
        # anticipated: ['watson-tts', 'google-tts', 'rec-embed', 'rec-service']
        response_dict['speaker'] = 'rec-service'
        response_dict['patient'] = 21
        # anticipated: ['michael', 'michaelV3'] (if watson-tts) default: michaelV3
        #          or: ['vlad', 'michaelV3'] (if rec-service) default: vlad
        response_dict['avatar'] = 'vlad'
        response_dict['oobhack'] = True;
        known_client = client_type == 'iOS' or client_type == 'Quest'
        if (known_client and setup == 'default'):
            pass
        elif (known_client and setup == 'lep'):
            response_dict['speaker'] = 'rec-service'
            response_dict['patient'] = 210
            response_dict['avatar'] = 'vlad'
        elif (known_client and setup == 'test'):
            response_dict['speaker'] = 'watson-tts'
            response_dict['avatar'] = 'michaelV3'
        elif (known_client and setup == 'test2'):
            response_dict['speaker'] = 'watson-tts'
            response_dict['avatar'] = 'michael'
        response_str = json.dumps(response_dict, indent=2)
        response = Response(response = response_str,
                            status = 200,
                            mimetype = 'application/json')
        return response
    else:
        abort(400)

@app.route("/score/", methods=['GET'])
def show_score():
    if 'convo_num' in request.args and 'secret' in request.args:
        num = request.args.get('convo_num','')
        secret = request.args.get('secret','')
        check_sql = ''' SELECT Convo_num, Uuid FROM RConversations
                        WHERE Convo_num = %s '''
        cursor = db.connection.cursor()
        check_num = None
        check_secret = None
        try:
            cursor.execute(check_sql, [num])
            record = cursor.fetchone()
            check_num = record[0]
            check_secret = record[1]
        except:
            db.connection.rollback()
        cursor.close()
        if (check_num is not None 
            and check_secret is not None 
            and check_secret == secret):
            return render_template("score.html", num=num)
        else:
            abort(404)
    else:
        abort(400)

@app.route("/Scores/<string:filename>", methods=['GET'])
def get_score_file(filename):
    try:
        return send_from_directory("cs_scores", filename, as_attachment=False, mimetype="text/plain")
    except FileNotFoundError:
        abort(404)


@app.route("/conversations/", methods=['POST'])
def conversations():
    if request.method == 'POST':
        ## TODO error handling:
        ##      error if not json
        ##      error if db fails
        logger.info(request.headers)
        logger.info(request.data)
        inputs = request.get_json()
        convo_num = -1
        convo_uuid = str(uuid.uuid4())
        inputs['ws_v'] = WS_VERSION
        inputs['uuid'] = convo_uuid
        if not 'group' in inputs:
            if conf['service_pipeline'] == 'cs_only':
                inputs['group'] = 'control'
            elif conf['service_pipeline'] == 'cs_rnn':
                inputs['group'] = 'test'
            elif conf['service_pipeline'] == 'random':
                inputs['group'] = 'control' if random.random() < 0.5 else 'test'
            else:
                abort(500)
            ins_sql = '''INSERT INTO RConversations (Client_ID, WS_Version, First_name, Last_name, Patient_choice, Input_method, Mic, Exp_group, Uuid)
                     VALUES (%(client)s, %(ws_v)s, %(first)s, %(last)s, %(patient)s, %(input)s, %(mic)s, %(group)s, %(uuid)s);'''
        num_sql = '''SELECT LAST_INSERT_ID();'''
        cursor = db.connection.cursor()
        error = False
        try:
            
            cursor.execute(ins_sql, inputs)
            db.connection.commit()
            cursor.execute(num_sql)
            convo_num = cursor.fetchone()[0]
        except:
            error = True
            db.connection.rollback()
        cursor.close()
        cs_greeting = cs_exchange(inputs['first'], inputs['last'], inputs['patient'], "")
        response_dict = {}
        status = 201
        headers = {}
        if not error:
            headers['location'] = "/conversations/" + str(convo_num) + "/"
            response_dict['status'] = 'ok'
            response_dict['resource'] = url_for('show_conversation', num=convo_num)
            response_dict['conversation_num'] = convo_num
            response_dict['greeting'] = cs_greeting
            response_dict['uuid'] = convo_uuid 
        else:
            status = 500
            response_dict['status'] = 'error'
            response_dict['resource'] = ''
            response_dict['conversation_num'] = ''
            response_dict['greeting'] = ''
            response_dict['uuid'] = ''

        response_str = json.dumps(response_dict, indent=2) + "\n"
        response = Response(response = response_str,
                            status = status,
                            headers = headers,
                            mimetype = 'application/json')
        return response

@app.route("/conversations/<int:num>/")
def show_conversation(num):
    response_dict = {}
    get_sql = '''SELECT * FROM RConversations
                 WHERE Convo_num = %s'''
    cursor = db.connection.cursor()
    try:
        cursor.execute(get_sql, [str(num)])
        record = cursor.fetchone()
        response_dict['num'] = record[0]
        response_dict['client'] = record[1]        
        response_dict['webservice_version'] = record[2]
        response_dict['first'] = record[3]
        response_dict['last'] = record[4]
        response_dict['patient'] = record[5]
        response_dict['input'] = record[6]
        response_dict['mic'] = record[7]
        response_dict['group'] = record[8]
        response_dict['raw_score'] = record[9]
    except Exception as e:
        logging.info(e)
        response = Response(response = '{"error": "not found"}',
                            status = 404,
                            mimetype = 'application/json')
        return response
    cursor.close()
    response_str = json.dumps(response_dict, indent=2)
    response = Response(response = response_str,
                        status = 200,
                        mimetype = 'application/json')
    return response

@app.route("/conversations/<int:convo_num>/query/", methods=['POST'])
def new_query(convo_num):
    if request.method == 'POST':
        ## TODO error handling:
        ##      error if not json
        ##      error if db fails
        ##      error if conversation does not exist

        error = False
        usr_first = ""
        usr_last = ""
        last_qnum = 0
        group = ""
        usr_sql = '''SELECT First_name, Last_name, MAX(Query_num), Exp_group, Patient_choice
                     FROM RConversations JOIN RQueries
                     ON RConversations.Convo_num = RQueries.Convo_num
                     WHERE RConversations.Convo_num = %s;'''
        cursor = db.connection.cursor()
        try:
            cursor.execute(usr_sql, [str(convo_num)])
            record = cursor.fetchone()
            usr_first = record[0]
            usr_last = record[1]
            group = record[3]
            patient = record[4]
            if record[2] is not None:
                last_qnum = int(record[2])
            else:
                last_qnum = 0
        except BaseException as e:
            ## TODO: needs work
            error = True
            logger.error(str(e))
            return ("Error connecting to database", 500)
        # IMPORTANT NOTE!!! this (new_qnum) is how uniqueness of query keys is maintained in the DB.
        # I'm sure this is a bad idea somehow, but that's how it's happening now.
        new_qnum = last_qnum + 1 
        inputs = request.get_json()
        logger.info(inputs)
        ## ask ChatScript
        to_cs = "[ q: " + str(new_qnum) + " ] " + inputs['query']
        cs_init_reply = cs_exchange(usr_first, usr_last, patient, to_cs)
        logger.info(cs_init_reply)
        if "score me" in inputs['query'] and cs_init_reply is not None:
            score_ins_sql = '''UPDATE RConversations SET Raw_score = %(raw_score)s
                               WHERE Convo_num = %(convo_num)s'''
            sql_data = {}
            sql_data['raw_score'] = cs_init_reply
            sql_data['convo_num'] = convo_num
            cursor = db.connection.cursor()
            try:
                cursor.execute(score_ins_sql, sql_data)
                db.connection.commit()
            except:
                logging.info("Database error")
                db.connection.rollback()
            cursor.close()
            
            response_dict = {}
            response_dict['status'] = 'ok'
            response_dict['resource'] = url_for('show_conversation', num=convo_num)
            response_dict['reply'] = ''
            response_str = json.dumps(response_dict, indent=2) + "\n"
            response = Response(response = response_str,
                                status = 201,
                                mimetype = 'application/json')
            return (response)
        why = cs_exchange(usr_first, usr_last, patient, ":why")
        try:
            cs_class = re.findall(r'u:(.*?) \(', why)[0].strip().lower()
        except:
            cs_class = "none" 
        try:
            cs_interp = cs_rnn_map[cs_class]
            cs_num = class_num(cs_interp)
        except KeyError:
            cs_interp = " "
            cs_num = None
        logger.info(cs_interp)
        if group == 'test' and len(inputs['query'].split()) > 2 :
            ## ask RNN
            probs, confs = rnn_predict(inputs['query'])
            #print(probs.data)
            best = best_idx(probs)
            rnn_interp = query(best)
            # rnn_reply = "NULL"
            unlog = torch.exp(probs)
            if cs_num is not None:
                cs_logprob = probs[0].tolist()[cs_num] 
            else:
                cs_logprob = None
            feats = compile_feats(probs.tolist()[0][best],
                                  entropy(unlog.data[0].numpy()),
                                  confs.tolist()[0][best],
                                  cs_logprob,
                                  rnn_interp)
            #print(feats)
            use_rnn = decider.switch_to_RNN(feats)
            #print(use_rnn)
            cs_re_reply = "NULL"
            if use_rnn:
                new_query = "[ q: " + str(new_qnum) + " ] " + rnn_interp
                cs_re_reply = cs_exchange(usr_first, usr_last, patient, new_query)
                reply = cs_re_reply
            else:
                reply = cs_init_reply
        else:
            reply = cs_init_reply
            use_rnn = False
            rnn_interp = None
            cs_re_reply = None
        ## set SQL input values
        #pdb.set_trace()
        ins_data = {}
        ins_data['convo_num'] = convo_num
        ins_data['query_num'] = new_qnum
        ins_data['query'] = inputs['query']
        ins_data['cs_interp'] = cs_interp
        ins_data['rnn_interp'] = rnn_interp
        ins_data['cs_init_reply'] = cs_init_reply
        ins_data['cs_retry_reply'] = cs_re_reply
        ins_data['choice'] = 'rnn' if use_rnn else 'cs'
        ins_sql = '''INSERT INTO RQueries 
                     (Convo_num, Query_num, Input_text, CS_interp, RNN_interp, CS_init_reply, CS_retry_reply, Choice)
                     VALUES (%(convo_num)s, %(query_num)s, %(query)s, %(cs_interp)s, %(rnn_interp)s, %(cs_init_reply)s, %(cs_retry_reply)s, %(choice)s);'''
        try:
            cursor.execute(ins_sql, ins_data)
            db.connection.commit()
        except:
            error = True
            db.connection.rollback()
        cursor.close()
        response_dict = {}
        status = 201
        headers = {}
        if not error:
            headers['location'] = "/conversations/" + str(convo_num) + "/query/" + str(new_qnum) + "/"
            response_dict['status'] = 'ok'
            response_dict['resource'] = url_for('show_conversation', num=convo_num) + "query/" + str(new_qnum) + "/" ##FIXME (implement GET)
            #response_dict['conversation_num'] = convo_num
            response_dict['reply'] = reply
        else:
            status = 500
            response_dict['status'] = 'error'
            response_dict['resource'] = ''
            #response_dict['conversation_num'] = ''
            response_dict['reply'] = ''

        response_str = json.dumps(response_dict, indent=2) + "\n"
        response = Response(response = response_str,
                            status = status,
                            headers = headers,
                            mimetype = 'application/json')

        return response #redirect(url_for('show_conversation', num=convo_num))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['zip', 'gz', 'wav']
    
@app.route("/conversations/<int:convo_num>/query/<int:query_num>/audio", methods=['POST'])
def add_audio(convo_num, query_num):
    if request.method == 'POST':
        logger.info(request.headers)
#        logger.info(request.data)
        status = 400
        response_dict = {}
        response_dict['status'] = 'error'
        response_dict['resource'] = ''
        response_dict['info'] = 'No file given'
        headers = {}
        # check if the post request has the file part
        if 'file' not in request.files:
            response_str = json.dumps(response_dict, indent=2) + "\n"
            return Response(response = response_str,
                            status = status,
                            headers = headers,
                            mimetype = 'application/json')
        audio = request.files['file']
        # could get an empty filename if submitted without a selection
        if audio.filename == '':
            response_str = json.dumps(response_dict, indent=2) + "\n"
            return Response(response = response_str,
                            status = status,
                            headers = headers,
                            mimetype = 'application/json')
        # check if query resource exists and no audio already
        ret_qnum = ""
        ret_audpath = ""
        select_sql = '''SELECT Query_num, Audio_path
                        FROM RQueries
                        WHERE Convo_num = %s
                        AND Query_num = %s;'''
        cursor = db.connection.cursor()
        try:
            cursor.execute(select_sql, [str(convo_num), str(query_num)])
            record = cursor.fetchone()
            if record:
                ret_qnum = record[0]
                ret_audpath = record[1]
            else:
                # if we get here, there is no query with that conv/query num.
                # TODO: could create the resource and wait for the query instead of failing
                status = 409 # Conflict
                response_dict['info'] = 'No corresponding query'
                response_str = json.dumps(response_dict, indent=2) + "\n"
                return Response(response = response_str,
                                status = status,
                                headers = headers,
                                mimetype = 'application/json')
        except:
            return ("Error connecting to database", 500)
        # print(ret_qnum, ret_audpath)
        ok_to_add_audio = (ret_qnum != "" and ret_audpath is None)
        if (not ok_to_add_audio):
            response_dict['info'] = 'Audio already exists'
        if audio and allowed_file(audio.filename) and ok_to_add_audio:
            filename = secure_filename(audio.filename)
            new_audio_path = os.path.join(conf['service_audio_storage'], filename)
            audio.save(new_audio_path)
            ins_sql = ''' UPDATE RQueries SET Audio_path = %s
                          WHERE Convo_num = %s AND Query_num = %s
                      '''
            try:
                cursor.execute(ins_sql, [new_audio_path, str(convo_num), str(query_num)])
                db.connection.commit()
            except:
                db.connection.rollback()
                
            status = 201
            response_dict['status'] = 'ok'
            response_dict['resource'] = url_for('show_conversation', num=convo_num) + "query/" + str(query_num) + "/audio" ##FIXME (implement GET)
            response_dict['info'] = ''
        
        cursor.close()
        response_str = json.dumps(response_dict, indent=2) + "\n"
        return Response(response = response_str,
                        status = status,
                        headers = headers,
                        mimetype = 'application/json')
    
    
if __name__ == "__main__":
    if conf['service_key'] is not None:
        http = WSGIServer((conf['service_host'], conf['service_port']),
                          app.wsgi_app,
                          keyfile=conf['service_key'],
                          certfile=conf['service_cert'],
                          log = logger,
                          error_log = logger)
    else:
        http = WSGIServer((conf['service_host'], conf['service_port']),
                          app.wsgi_app,
                          log = logger,
                          error_log = logger)        
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        http.stop()
        exit(signal.SIGTERM)

    
