
import os
import logging

TWITTER_KEYS = '~/.ssh/twitter'
EMAIL_SERVER = 'localhost'
EMAIL_SENDER = 'reflfit@reflectometry.org'

def notify(user, msg, body="No body", level=1):
    """
    Send a notfication message to the user regarding the job status.
    """

    if not user:
        pass
    elif user.startswith('@'):
        tweet(user, msg)
    elif '@' in user:
        email(EMAIL_SENDER, [user], body, subject=msg, server=EMAIL_SERVER)
    else:
        logging.debug("%s : %s"%(user, msg))

twitter = None
def tweet(user, msg):
    global twitter
    if twitter is None: twitter = open_twitter(TWITTER_KEYS)
    twitter.direct_messages.new(user=user, text=msg)

def open_twitter(authfile):
    from twitter import Twitter, OAuth
    for line in open(os.path.expanduser(authfile)).readlines():
        exec line
    auth = OAuth(access_token, access_secret,
                 consumer_key, consumer_secret)
    return Twitter(auth=auth)


def email(sender, receivers, message, subject='no subject', server='localhost'):
    """
    Send an email message to a group of receivers
    """
    import smtplib

    if ':' in server:
        host,port = server.split(':')
        port = int(port)
    else:
        host,port = server,25
    header="From: %s\r\nTo: %s\r\nSubject: %s\r\n\r\n"
    header %= sender,", ".join(receivers),subject
    #print "Sending the following mail message:\n"+header+message
    #print "Trying to connect to",host,port
    smtp = smtplib.SMTP(host,port)
    #print "Connection established"
    smtp.sendmail(sender,receivers,header+message)
    #print "Mail sent from",sender,"to",", ".join(receivers)
    smtp.quit()


if __name__ == "__main__":
    msg = 'test notification'
    body = 'See http://reflectometry.org for details.'
    #notify('@pkienzle', msg, body)
    notify('paknist@gmail.com', msg, body)
