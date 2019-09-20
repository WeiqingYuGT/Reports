from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, UserDefinedFunction
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime
from string import Template
import subprocess

def initEnv(taskname=None):
    if taskname is not None:
        spark = SparkSession.\
                builder.\
                appName(taskname).\
                enableHiveSupport().\
                getOrCreate()
    else:
        spark = SparkSession.\
                builder.\
                enableHiveSupport().\
                getOrCreate()
    return spark

def getRecent(tbl,type='hive',max1=True):
    spark = initEnv()
    if type=='hive':
        db, name = tbl.split('.')
        return db+'.'+spark.sql("show tables in {}".format(db)).\
            filter(col("tableName").like(name+'%')).\
            agg(F.max(col('tableName'))).collect()[0][0]
    elif type=='hdfs':
        dirs = subprocess.check_output('hdfs dfs -find '+tbl,shell=True).split()
        if max1:
            return max(dirs)
        else:
            return min(dirs)

udf_fill = UserDefinedFunction(lambda x: 'unknown' if x is None or x=="" else x, StringType())

def loadScienceCoreData(dt, logger, prod_types = ('exchange','display') ,
                        filters_pub=True, filters_other = '',fill=True,
                        selected = ('uid','request_id')):
    spark = initEnv()
    schema1 = spark.read.format('orc'). \
        load("s3a://xad-science/dw/science_core_ex/us/display/{}/10/fill/pos/". \
             format(datetime.strptime(dt, '%Y-%m-%d').strftime('%Y/%m/%d'))).schema

    path_template = Template("s3a://xad-science/dw/science_core_ex/us/{${prods}}/${dt}/{${hrs}}/{${fill}}/{${loc}}")

    logger.info(','.join(prod_types))
    paths = path_template.substitute(dt = datetime.strptime(dt, '%Y-%m-%d').strftime('%Y/%m/%d'),
                                     prods = ','.join(prod_types),
                                     hrs = ','.join(['0'+str(x) if x<10 else str(x) for x in range(24)]),
                                     loc = 'pos,tll,rest',fill='fill' if fill else 'fill,nf')
    logger.info('running with the following filters: {}'.format(filters_other))
    logger.info('columns being selected: {}'.format(selected))
    dat1 = spark.read.format('orc'). \
        load(paths, schema=schema1)
    if filters_pub:
        pub_df = spark.read.option('delimiter','\t').\
            csv('hdfs:///apps/hive/warehouse/canliang.db/privacy_publisher_mode_2/publisher_mode_2')
        pub_df = pub_df.withColumn('id',col('_c0'))
        dat1 = dat1.join(pub_df,dat1.pub_id==pub_df.id,how='left').\
            where('id is null').\
            drop(pub_df.id)
    if len(filters_other)>0:
        dat1 = dat1.where(filters_other). \
            select(selected)
    else:
        dat1 = dat1.select(selected)
    return dat1
