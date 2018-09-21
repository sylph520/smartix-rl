import mysql.connector
from agent import Agent


class Action:

    def __init__(self):
        self.current_state = list()
        self.agent = Agent()
        self.map_indexes = self.agent.reset_map_indexes()

    def add_index(self, column):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        for line in open('../name_of_indexes'):
            command = ('CREATE INDEX %s ON lineitem (%s);\n' % (line.replace('\n', ''), column))
            cursor.execute(command)

    def drop_index(self, index):
        cnx = mysql.connector.connect(host='127.0.0.1', user='root', passwd='teste', db='tpch')
        cursor = cnx.cursor(buffered=True)
        command = ('DROP INDEX %s ON lineitem;' % str(index))
        cursor.execute(command)
