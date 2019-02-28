from nltk.translate.bleu_score import sentence_bleu
import jieba
import Campa_SqlActivater
import pandas as pd
import json
import numpy as np
import logging
import fastText.FastText  as ff
from random import sample
import re
import os
from collections import Counter
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix

# 设置plan_id
plan_id = 7
key_word = '阿强'

# 连接数据库
SqlQueryActivater = Campa_SqlActivater.SqlQueryActivater()
conn_for_apis_mysql = SqlQueryActivater.get_conn_for_apis_mysql()
conn_for_analyze_mysql = SqlQueryActivater.get_conn_for_analyze_mysql()
conn_for_apis_postgresql = SqlQueryActivater.get_conn_for_apis_postgresql()
conn_for_book_mysql = SqlQueryActivater.get_conn_for_book_mysql()
conn_for_static_mysql = SqlQueryActivater.get_conn_for_static_mysql()

# 加载数据
sql_qi_data = '''
    SELECT
        panda_challenge_records.plan_id,
        manager_assignment_job_records.problem_id,
        manager_correct_records.job_id,
        panda_challenge_records.answer,
        manager_correct_records.correct_text,
        manager_correct_records.`status`
    FROM
        manager_correct_records
    INNER JOIN manager_assignment_job_records ON manager_correct_records.job_id = manager_assignment_job_records.id
    INNER JOIN panda_challenge_records ON manager_assignment_job_records.record_id = panda_challenge_records.id
    WHERE
        panda_challenge_records.plan_id = {0}
    AND manager_correct_records.`status` >= 2
    AND manager_correct_records.correct_text <> ''
    and manager_correct_records.correct_text LIKE '%{1}%'
'''.format(plan_id, key_word)
qi_data = pd.read_sql(sql_qi_data, conn_for_apis_mysql)


# 提取学员的作业答案和批改员的批改答案，并且拼接在一起，然后进行分词，若是测试数据，则直接予以剔除
def create_corpus(index):
    try:
        try:
            this_correct_text = json.loads(qi_data['correct_text'][index])
            this_correct_text = this_correct_text[0]['admin_answer']
        except:
            this_correct_text = qi_data['correct_text'][index]
        # 去掉换行符和tab符
        this_correct_text = re.sub('\n|\t|<br>| ', '', this_correct_text)
        this_correct_text = re.sub('[，。""“”：;；:‘’（）《》、_,?？!！]', '', this_correct_text)
        # 进行中文分词
        this_correct_text = jieba.cut(this_correct_text)
        this_correct_text = ' '.join(this_correct_text)
        return this_correct_text
    except:
        return ''


qi_data = qi_data.reset_index()
qi_data['corpus'] = qi_data['index'].apply(create_corpus)
qi_data = qi_data[(qi_data['corpus'] != '')]
qi_data = qi_data.sort(['job_id', 'status'], axis=0)
qi_data = qi_data.drop_duplicates(['job_id'])


# 选取部分优秀的评语作为参考评语
def create_reference(select=50):
    qi_data_good = qi_data[(qi_data['status'] == 3)]
    qi_data_good = qi_data_good.sample(select, replace=False)
    reference = []
    for i in qi_data_good['corpus']:
        reference.append(i.split(' '))
    return reference

reference = create_reference(select=3)

reference_demo='关键信息写的精简详细，看的出信息梳理你已经掌握的很不错了，结尾还客气的表达谢意，很棒。但是开头最好要表明请求的原因，中间一些不重要的信息可以适当删除，例如“我早上7:40坐车从遂宁到成都，再转车到成都东站，你帮我买好票我好直接到东站去取票。”这句话完全可以删除。虽然有点小瑕疵，但是你已经很棒了，我们的课程刚刚开始，继续加油哦。课程和作业内容记得要及时温习，举一反三，并在工作和生活中进行实践哦。相信你一定会越来越优秀的。'
reference_demo=jieba.cut(reference_demo)
reference_demo=list(reference_demo)
reference.append(reference_demo)

# 计算BLEU值
def compute_bleu(text):
    text = text.split(' ')
    score = sentence_bleu(reference, text,weights=(0,0,0,1))
    return score

qi_data['bleu'] = qi_data['corpus'].apply(compute_bleu)

# 划分训练集和测试集
qi_data['label'] = 0
qi_data['label'][(qi_data['status'] == 3)] = 1
qi_data_train = qi_data.sample(int(qi_data.shape[0] * 0.8), replace=False)
qi_data_test = qi_data[~(qi_data['index']).isin(qi_data_train['index'])]

# 利用BLEU值来建立分类模型
logistic_model = LR()
logistic_model.fit(qi_data_train[['bleu']], qi_data_train['label'])
logistic_model.score(qi_data_test[['bleu']], qi_data_test['label'])  # 准确率

# 计算测试集的混淆矩阵
pre = logistic_model.predict(qi_data_test[['bleu']])
confusion_matrix(qi_data_test['label'],pre)

