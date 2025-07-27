from imapclient import IMAPClient
import pyzmail
from dotenv import load_dotenv
import os


class Email():
    def __init__(self):    
        load_dotenv()
        self.EMAIL = os.getenv('EMAIL')
        self.APP_PASSWORD = os.getenv('APP_PASSWORD')
        self.HOST = os.getenv('HOST')

    def get_latest_n_emails(self,n = 10):
        with IMAPClient (self.HOST) as client:
            client.login(self.EMAIL,self.APP_PASSWORD)
            client.select_folder("INBOX",readonly=True)


            messages = client.search(['NOT','Deleted'])
            # print(f"Found {len(messages)} messages")

            latest_messages = messages[-n:]
            self.data = []

            for uid in latest_messages:
                raw_message = client.fetch([uid], ['BODY[]', 'FLAGS'])

                message = pyzmail.PyzMessage.factory(raw_message[uid][b'BODY[]'])
                subject = message.get_subject()
                from_ = message.get_addresses('from')[0][1]
                if message.text_part:
                    body = message.text_part.get_payload().decode(message.text_part.charset)

                else:
                    body = 'No body found'


                self.data.append(( from_, subject, body))  

    def find_next_email(self,n=0, max_char = 10000):
        return self.data[n][0],self.data[n][1],self.data[n][2][:max_char]

