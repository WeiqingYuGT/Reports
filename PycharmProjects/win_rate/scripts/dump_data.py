from datetime import datetime, timedelta
from utils.pyspark_helper import initEnv
from config.properties import daily_path, lookback_window, n_split, dump_loc, retension_days
import pyspark.sql.functions as F
import os
import boto3

class dumpData:
    def __init__(self,logger,dt=''):
        if dt=='':
            dt1 = datetime.today().strftime('%Y-%m-%d')
            s3 = boto3.client('s3')
            while 'Contents' not in s3.list_objects(Bucket='xad-science',
                                Prefix=daily_path[18:]+'dt={}'.format(dt1)):
                dt1 = (datetime.strptime(dt1,'%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d')
            self.dt = dt1
        else:
            self.dt = dt
        self.data = self._loadDat()
        self.data.cache()
        self.logger = logger

    def _loadDat(self,lookback = lookback_window):
        spark = initEnv()
        sdt = (datetime.strptime(self.dt,'%Y-%m-%d')-timedelta(days=lookback)).\
            strftime("%Y-%m-%d")
        data = spark.read.orc(daily_path).\
            where("""dt>='{}'""".format(sdt))
        return data

    def _splitTVT(self):
        maxDt = self.data.select(F.max('dt')).collect()[0][0]
        valDt = (datetime.strptime(maxDt,'%Y-%m-%d')-timedelta(days=1)).strftime('%Y-%m-%d')
        trn = self.data.where("""dt<'{}'""".format(valDt))
        val = self.data.where("""dt='{}'""".format(valDt))
        tst = self.data.where("""dt='{}'""".format(maxDt))
        return trn, val, tst

    def splitWrite(self,splits = n_split):
        ratio = 1.0/n_split*1.5
        trn, val, tst = self._splitTVT()
        os.system('rm -r {}train/*.csv'.format(dump_loc))

        if splits==1:
            for dt1 in [x[0] for x in trn.select('dt').distinct().toPandas().values.tolist()]:
                trn.where("""dt='{}'""".format(dt1)).toPandas().\
                    to_csv(dump_loc+'train/train_data_{}.csv'.format(dt1),index=False)
            self.logger.info('Finished writing '+dump_loc+'train/train_data.csv')
        else:
            for i in range(splits):
                os.system('mkdir {}train/sample={}'.format(dump_loc,i))
                trn.sample(False, ratio, seed=1+i*100).toPandas().\
                    to_csv(dump_loc+'train/sample={}/train_data.csv'.format(i),index=False)
                self.logger.info('Finished writing '+dump_loc+'train/sample={}/train_data.csv'.format(i))
        val.toPandas().to_csv(dump_loc+'validation/validation_data.csv',index=False)
        self.logger.info('Finished writing ' + dump_loc+'validation/validation_data.csv')
        tst.toPandas().to_csv(dump_loc+'test/test_data.csv',index=False)
        self.logger.info('Finished writing ' + dump_loc+'test/test_data.csv')
        os.system('sudo cp -r {}/* /home/weiqingyu/win_rate/data/'.format(dump_loc))

    def cleanData(self, retension=retension_days):
        lastDt = datetime.today()-timedelta(days=retension)
        s3 = boto3.client('s3')
        flist = s3.list_objects(Bucket='xad-science',Prefix=daily_path[18:])
        firstDt = min(flist['Contents'], key=lambda x: x['Key'])['Key'].split('/')[2][3:]
        firstDt = datetime.strptime(firstDt,'%Y-%m-%d')
        while firstDt<lastDt:
            self.logger.info('Clearing old data on {}'.format(firstDt))
            bucket = boto3.resource('s3').Bucket('xad-science')
            bucket.objects.filter(Prefix=daily_path[18:]+'dt='+firstDt.strftime('%Y-%m-%d')).delete()
            firstDt = firstDt+timedelta(days=1)
        return