from pg_database import PG_Database


class Action:


    def __init__(self, table, column, type):
        # Database instance
        # self.db = PG_Database()

        # Action attributes
        self.table = table
        self.column = column
        self.type = type


    def __repr__(self):
        return str(self.table) + ',' + str(self.column) + ',' + str(self.type)


    def __hash__(self):
        return hash(str(self))


    def __eq__(self, other):
        return str(self) == str(other)


    def execute(self, db):
        if self.type == 'DROP':
            db.drop_index(self.table, self.column)
        elif self.type == 'CREATE':
            db.create_index(self.table, self.column)
        else:
            print("PASS action!")



if __name__ == "__main__":
    action1 = Action("lineitem", "l_shipmode", "CREATE")
    action2 = Action("lineitem", "l_shipmode", "DROP")
    action1.execute()
    action2.execute()
