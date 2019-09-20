from utils import initEnv, logHelper, udf_fill
from config import feature_list, feature_list_bk, target_set, neptune_vendor_list
import pyspark.sql.functions as F
from pyspark.sql.functions import UserDefinedFunction, col
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
import numpy as np

class runModel:

    def __init__(self, pub_id, logger, bk_feat=False):
        self.pub_id = pub_id
        self.bk_feat = bk_feat
        self.logger = logger

    def _featProc(self,feat):
        if feat in ('adv_bid_rates','pub_bid_rates'):
            udf1 = UserDefinedFunction(lambda x: x[8:-2], StringType())
            return udf1(col(feat)).cast('double').alias(feat)

    def _cateProc(self, feat, dat, vec=False, trim=True):
        if dat.agg(F.countDistinct(col(feat))).collect()[0][0] == 1:
            dat = dat.withColumn(feat, F.lit(0))
            return dat

        if trim:
            df1 = dat.groupby(feat).count().toPandas().sort_values('count', ascending=False)
            df1['cdf'] = np.cumsum(df1['count'] / sum(df1['count']))
            df1 = df1.reset_index(drop=True)
            feat_set = (df1[feat][0:(1 + min(df1[df1['cdf'] > 0.80].index))]).tolist()
            udf1 = UserDefinedFunction(lambda x: 'unknown' if x not in feat_set else x, StringType())
            dat = dat.withColumn(feat + '_1', udf1(udf_fill(feat))).drop(feat + '_vec')
        else:
            feat_set = dat.select(feat).distinct().toPandas()[feat].values.tolist()
            dat = dat.withColumn(feat + '_1', udf_fill(feat)).drop(feat + '_vec')
            dat = dat.fillna('empt', feat + '_1')

        if vec:
            stringIndexer = StringIndexer(inputCol=feat + '_1', outputCol="categoryIndex")
            model = stringIndexer.fit(dat)
            indexed = model.transform(dat)
            encoder = OneHotEncoder(inputCol="categoryIndex", outputCol=feat + '_vec')
            dat = encoder.transform(indexed).drop('categoryIndex').drop(feat + '_1')
        else:
            for f in feat_set:
                dat = dat.withColumn(feat + '_' + str(f).replace('.', '_'), (col(feat) == f).cast('int'))
                dat = dat.drop(feat + '_1')
        return dat

    def _genSummary(self,dat,feat):
        if dat.dtypes[[x[0] for x in dat.dtypes].index(feat)][1]=='string':
            tmp = dat.groupby(feat)
            cnt = tmp.count().toPandas()
            ratio = tmp.agg(F.avg('ad_impression')).toPandas()
            cnt['count'] = cnt['count']/sum(cnt['count'])
            cnt.columns = [feat, 'perc']
            ratio.columns = [feat, 'avg_win_rate']
            return [cnt, ratio]
        else:
            return dat.stat.corr(feat, 'ad_impression')

    def run(self,thres):
        spark = initEnv()
        dat = spark.sql("select {} from weiqingyu.win_rate_reqs1 where pub_id = {} and ad_vendor_id in ({})".\
                        format(','.join(feature_list+target_set) if not self.bk_feat\
                               else ','.join(feature_list+feature_list_bk+target_set),
                               self.pub_id,','.join(map(str,neptune_vendor_list))))
        dat = dat.sample(False, thres)
        mr1 = dat.where('adv_bid_rates is null').count()*1.0/dat.count()
        mr2 = dat.where('pub_bid_rates is null').count()*1.0/dat.count()
        if mr1>0.1:
            self.logger.warn("Missing rate of adv_bid_rates for {}: {}".format(self.pub_id,mr1))
        if mr2>0.1:
            self.logger.warn("Missing rate of adv_bid_rates for {}: {}".format(self.pub_id,mr2))

        dat1 = dat.\
            where("adv_bid_rates is not null and pub_bid_rates is not null and xad_revenue is not null and pub_revenue is not null")
        dat1 = dat1.fillna(0,subset='ad_impression')
        dat1 = dat1.withColumn('adv_bid_rates',self._featProc('adv_bid_rates',dat1)).\
            withColumn('pub_bid_rates',self._featProc('pub_bid_rates',dat1))
        dat1.cache()



if __name__=='__main__':
    logger = logHelper('../logs/data_ETL.log').getlogger()
