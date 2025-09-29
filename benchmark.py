from pg_database import PG_Database

class Benchmark:


    def __init__(self, benchmark = "TPCH"):
        self.benchmark = benchmark
        if (self.benchmark == "TPCH"):
            wk_file = 'data/workload/tpch_custom_20.sql'
            with open(wk_file, 'r') as f:
                content = f.read()
                queries = content.split('\n')
                if queries[-1] == '':
                    queries.pop()
            self.queries =  queries
        else:
            raise f"No corresponding benchmark {benchmark}"


    def run(self, pg_database:PG_Database):
        wk_cost = pg_database.get_workload_cost(self.queries)
        return wk_cost
