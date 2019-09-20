from collections import defaultdict
import logging
import smtplib
# Import the email modules we'll need
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class dtClass:
    def __init__(self,dt):
        self.dt = dt
        self.datetime = datetime.strptime(dt,'%Y-%m-%d')
        self.y = str(self.datetime.year)
        self.m = str(self.datetime.month) if self.datetime.month>9 else '0' + str(self.datetime.month)
        self.d = str(self.datetime.day) if self.datetime.day>9 else '0' + str(self.datetime.day)
    def toStr(self,format):
        return self.datetime.strftime(format)

# Create message container - the correct MIME type is multipart/alternative.
msg = MIMEMultipart('alternative')


to_list = ['weiqing.yu@groundtruth.com']
sender = 'weiqing.yu@groundtruth.com'
COMMASPACE = ', '


# Send the message via our own SMTP server, but don't include the
# envelope header.

def _send_msg(subject, text):
    plain = MIMEText(text, 'plain')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = COMMASPACE.join(to_list)
    msg.attach(plain)
    s = smtplib.SMTP('xad-com.mail.protection.outlook.com')
    s.sendmail(sender, to_list, msg.as_string())
    s.quit()

def send_msg(subject, text, done_items=None):
    NEWLINE = ''
    body = text + '\n' + 'Finished tasks:\n'
    body += '' if done_items is None else NEWLINE.join(done_items)
    _send_msg(subject, body)

class logHelper:
    def __init__(self,logger_name):
        self.name = logger_name
    def getlogger(self,**keys):
        if len(keys)==0:
            logging.basicConfig(
                format = '-- [%(asctime)s - %(levelname)s]\t %(message)s',
                datefmt = '%Y%m%d %I:%M%S%p',
                filename = self.name
            )
        else:
            logging.basicConfig(
                filename = self.name,
                **keys
            )
        logging.root.level = logging.INFO
        lg = logging.getLogger()
        return lg

def mapBucketNames(kvList):
    # type: (str) -> dict
    mappingDict = defaultdict(lambda: 'Unknown')
    for kv in kvList:
        splitKV = kv.split("-")
        keys = [x.strip() for x in splitKV[0].split(",")]
        value = splitKV[1].strip()
        for key in keys:
            mappingDict[key] = value
    return mappingDict

