from sklearn.externals import joblib

a = joblib.load('../data/data_seq.pkl')
joblib.dump(a,'../data/data_seq.dat',protocol=2)
b = joblib.load('../data/query_dict_seq.pkl')
joblib.dump(b,'../data/query_dict_seq.dat',protocol=2)