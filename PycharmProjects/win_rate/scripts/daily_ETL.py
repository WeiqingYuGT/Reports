from pyspark.sql.functions import UserDefinedFunction, col
from pyspark.sql.types import *
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from utils.pyspark_helper import loadScienceCoreData
from config.properties import feature_list, target_set, sample_rate, daily_path, neptune_vendor_list
from utils.common import logHelper

class loadData:
    def __init__(self, dt, logger):
        self.dt = dt
        self.logger = logger

    def _loadSrcDat(self):
        return loadScienceCoreData(self.dt,self.logger,selected=feature_list+target_set+['adgroup_id'],
                                   filters_other='ad_vendor_id in ({})'.format(','.join(map(str,neptune_vendor_list)))).\
            sample(False,sample_rate)

    def writeDat(self):
        data = self._loadSrcDat()
        self.logger.info('finished read the data schema')
        data.write.mode('overwrite').orc(daily_path+'dt={}/'.format(self.dt))
