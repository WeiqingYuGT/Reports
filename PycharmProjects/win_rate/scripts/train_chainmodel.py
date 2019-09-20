import numpy as np
import pandas as pd
from config.properties import dump_loc, feature_to_use, bid_price_lower_bound
from config.properties import upload_location, chain_model_location, deliver_model_location
from config.properties import n_estimators, max_depth, n_jobs, min_impurity_decrease, max_features, bid_price_quantiles
import glob
from datetime import datetime
from mars.models import BinaryRandomForestClassifier, IsotonicRegression, ChainModel
import json
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import os


class trainChain:
    def __init__(self, logger):
        self.logger = logger
        self.trn, self.val, self.tst = self._loadTVT()
        self.trn, self.val, self.tst = self._toHourWeekday(self.trn), \
                                       self._toHourWeekday(self.val), \
                                       self._toHourWeekday(self.tst)
        self.trn_feats = self._updateFeatList()
        self.runProcCate(self.trn)
#        self.runProcCate(self.val)
#        self.runProcCate(self.tst)
        self.chainmodel = None
        self.chainmodel_delivery = None
        self.enc = None

    def _procBidRate(self,df):
        df['pub_bid_rates'] = df.pub_bid_rates.str.slice(8,-2).astype(float)

    def _bucketizeBidRate(self, df, quants):
        df['pub_bid_rates'] = pd.cut(df.pub_bid_rates, quants, \
                        right = False, labels = list(map(str,range(len(quants)-1)))).\
                        astype(str)

    def _updateFeatList(self):
        str_feats = self.trn[feature_to_use].dtypes=='O'
        str_feats = str_feats[str_feats==True].index.tolist()
        num_feats = list(set(feature_to_use)-set(str_feats))
        int_feats = self.trn[num_feats].dtypes=='int'
        int_feats = int_feats[int_feats==True].index.tolist()
        float_feats = list(set(num_feats)-set(int_feats))
        return [str_feats , int_feats , float_feats]

    def _procCateFeat(self,coln,df,thres = 0.9,hard_thres = 1000, max_cate = 200):
        cnt = df.groupby(coln)[coln].count().sort_values(ascending=False)
        cumcnt = np.cumsum(cnt/sum(cnt))
        if sum(cumcnt<thres)<5:
            keys = cnt[cnt>hard_thres].index.tolist()
        else:
            keys = cumcnt[cumcnt<thres].index.tolist()
        keys = keys[0:max_cate]
        df.loc[~np.isin(df[coln],keys),coln] = 'other'
        return

    def runProcCate(self,df):
        self.logger.info('Started Processing categorical features')
        for f in self.trn_feats[0]:
            self._procCateFeat(f,df)
        self.logger.info('Done')
        return

    def _getFeatEng(self,coln,ftype,keys=None):
        if ftype=='OneHot':
            return {
                "type":ftype,
                "args":{
                    "column": 'wr_'+coln,
                    "values": keys
                }}
        elif ftype=='DirectNumber':
            return {"type": ftype,
                    "args": {
                        "column": 'wr_'+coln
                    }}
        else:
            raise Exception('feature engineer type not acceptable: {}'.format(ftype))

    def _toHourWeekday(self,df):
        df['hour'] = df.r_timestamp.apply(lambda x: datetime.fromtimestamp(int(x/1000)).hour)
        df['weekday'] = df.r_timestamp.apply(lambda x: datetime.fromtimestamp(int(x/1000)).weekday())
        df = df.drop('r_timestamp',axis=1)
        df = df.dropna(axis = 0)
        return df

    def _loadCSV(self,filepath):
        if "*" in filepath:
            files = glob.glob(filepath)
            dat = pd.read_csv(files[0])
            for f in files[1:]:
                dat = dat.append(pd.read_csv(f))
        else:
            dat = pd.read_csv(filepath)
        self._procBidRate(dat)
        dat = dat.loc[dat.pub_bid_rates>bid_price_lower_bound]
        return dat

    def _loadTVT(self):
        trn_dat = self._loadCSV(filepath = dump_loc+'train/*.csv')
        val_dat = self._loadCSV(filepath = dump_loc+'validation/validation_data.csv')
        tst_dat = self._loadCSV(filepath = dump_loc+'test/test_data.csv')
        if bid_price_quantiles is not None:
            quants = bid_price_quantiles
        else:
            quants = trn_dat.pub_bid_rates.quantile(q=[0,0.2,0.4,0.6,0.8,1])
        self._bucketizeBidRate(trn_dat,quants)
        self._bucketizeBidRate(val_dat,quants)
        self._bucketizeBidRate(tst_dat,quants)
        self.logger.info('finished loading raw train, validation and test data')
        return trn_dat, val_dat, tst_dat

    def _genConfig(self,df):
        model_config = []
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df[self.trn_feats[0]])
        for i in range(len(self.trn_feats[0])):
            model_config.append(self._getFeatEng(self.trn_feats[0][i],'OneHot',keys=enc.categories_[i].tolist()))

        for feat in self.trn_feats[1]:
            model_config.append(self._getFeatEng(feat,'DirectNumber'))

        for feat in self.trn_feats[2]:
            model_config.append(self._getFeatEng(feat,'DirectNumber'))
        return enc, model_config

    def dumpConfig(self,df,path):
        enc, feat_config = self._genConfig(df)
        model_config = {"type":"ChainModel",
                        "features":feat_config}
        with open(path+'model_config.json','w') as outfile:
            json.dump(model_config,outfile)
        self.logger.info('Finished dumping feature engineering configuration file')
        return enc, feat_config

    def _getOneHot(self,df,enc):
        dat = enc.transform(df[self.trn_feats[0]])
        dat = hstack([dat, df[self.trn_feats[1]]])
        dat = hstack([dat, df[self.trn_feats[2]]])
        dat = dat.fillna(0)
        return dat

    def caliPlot(self,pred, label, n_buck=20):
        cnt, cali, prob, interv = [], [], [], (pred.max() - pred.min()) / n_buck
        self.logger.info("Calibration INFO")
        for j in range(n_buck):
            i = j * interv + pred.min()
            ind = ((pred >= i) & (pred < i + interv))
            cnt.append(sum(ind))
            num1 = sum(label[ind]) / cnt[-1]
            cali.append(num1 if num1 != 0 or j == 0 else cali[-1])
            num = sum(pred[ind]) / cnt[-1]
            prob.append(num if num != 0 else i + interv / 2)
            self.logger.info("bucket {}: [{},{}]".format(i,cali[-1],prob[-1]))
        diag = [x * 0.1 for x in range(11)]
        return [0.] + cali + [1.], [2 * prob[0] - prob[1]] + prob + [2 * prob[-1] - prob[-2]]

    def runModelTraining(self):
        self.enc, feat_config = self.dumpConfig(self.trn, dump_loc+'models/')

        trn_dat = self._getOneHot(self.trn,self.enc)
        trn_lab = (~self.trn.winbid.isna())*1.0
        trn_dat = trn_dat.fillna(0)
        rf_model = BinaryRandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth, n_jobs=n_jobs, max_features = max_features,
                            min_impurity_decrease = min_impurity_decrease,
                            random_state = 0, verbose=1)
        rf_model.fit(trn_dat, trn_lab)

        val_pred = rf_model.predict_proba(self._getOneHot(self.val,self.enc))[:,1]
        val_lab = (~self.val.winbid.isna())*1.0

        iso_model = IsotonicRegression(y_min=0, y_max = 1,out_of_bounds='clip')
        cali, prob = self.caliPlot(val_pred, val_lab)
        iso_model.fit(prob, cali)

        self.chainmodel = ChainModel(tail_models = [rf_model], head_model = iso_model)
        self.chainmodel.cc_dump(chain_model_location,overwrite = True)
        self.logger.info('Finished dumping chain model')

        self.testMetric(self.tst,rf_model, iso_model)

        dev_dat = self._getOneHot(self.trn.loc[trn_lab==1],self.enc)
        dev_lab = self.trn.loc[trn_lab==1,'ad_impression']
        dev_lab = 1.0 * (dev_lab>0)

        deliver_model = BinaryRandomForestClassifier(n_estimators=int(n_estimators/2),
                        max_depth=max_depth, n_jobs=n_jobs, max_features = max_features,
                        min_impurity_decrease = min_impurity_decrease,
                        random_state = 0, verbose=1)
        deliver_model.fit(dev_dat, dev_lab)

        val_dev_dat = self._getOneHot(self.val.loc[val_lab==1],self.enc)
        val_dev_pred = deliver_model.predict_proba(val_dev_dat)[:,1]
        val_dev_lab = self.val.loc[val_lab==1,'ad_impression']
        val_dev_lab = 1.0 * (val_dev_lab>0)

        dev_iso_model = IsotonicRegression(y_min=0, y_max = 1,out_of_bounds='clip')
        dev_cali, dev_prob = self.caliPlot(val_dev_pred, val_dev_lab)
        dev_iso_model.fit(dev_prob, dev_cali)

        self.chainmodel_delivery = ChainModel(tail_models = [deliver_model], head_model = dev_iso_model)
        self.chainmodel_delivery.cc_dump(deliver_model_location, overwrite=True)
        self.logger.info('Finished dumping delivery model')

        self.testMetric(self.tst.loc[~self.tst.winbid.isna()],deliver_model, dev_iso_model, ctype='delivery')
        os.system('aws s3 cp --recursive {} {}'.format(dump_loc+'models/', upload_location))
        os.system('sudo rm -r data/test/')
        os.system('sudo rm -r data/train/')
        os.system('sudo rm -r data/validation/')
        return

    def testMetric(self,df1, rf_model, iso_model,ctype = 'win_rate', thres = 10):
        df = df1.copy()
        self.logger.info('Performance metrics')
        tst_pred = rf_model.predict_proba(self._getOneHot(df,self.enc))[:,1]
        tst_cali_pred = pd.DataFrame(tst_pred)[0].apply(lambda x: iso_model.predict([[x]])[0]).values
        if ctype=='win_rate':
            tst_lab = (~df.winbid.isna())*1.0
        else:
            tst_lab = (df.ad_impression>0)*1.0
        tst_cali, tst_prob = self.caliPlot(tst_cali_pred, tst_lab)
        df['pred'] = tst_cali_pred
        df['label'] = tst_lab
        metric = df.groupby('adgroup_id'). \
            agg({'pred': ['sum','count'], 'label': 'sum'}).reset_index()
        metric.columns = ['adgroup_id','pred','cnt','label']
        metric1 = metric.loc[metric.pred>thres]
        flip_error = (metric1.pred- metric1.label).abs()/np.maximum(1, np.minimum(metric1.pred,metric1.label))
        self.logger.info('Flip Error Mean: {}'.format(flip_error.mean()))
        self.logger.info('Flip Error Median: {}'.format(flip_error.median()))

