from utils.common import logHelper, send_msg
import argparse
import traceback


def p_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--type',choices=['daily','dump','train'])
    parser.add_argument('-d','--date',help='the date to process')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    logger = logHelper('./logs/runner.log').getlogger()
    args = p_args()
    try:
        if args.type=='daily':
            from scripts.daily_ETL import loadData
            from utils.pyspark_helper import initEnv
            spark = initEnv('win rate daily ETL')
            lD = loadData(args.date, logger)
            lD.writeDat()
            spark.stop()
        elif args.type=='dump':
            from scripts.dump_data import dumpData
            from utils.pyspark_helper import initEnv
            spark = initEnv('win rate dump training file')
            dD = dumpData(logger)
            dD.splitWrite()
            dD.cleanData()
            spark.stop()
        elif args.type == 'train':
            from scripts.train_chainmodel import trainChain
            tc = trainChain(logger)
            tc.runModelTraining()
    except Exception as e:
        errors = str(e)
        logger.warn("Error! /n{}/n{}".format(errors,traceback.format_exc()))
        send_msg('Win rate ',errors)
        if spark:
            spark.stop()