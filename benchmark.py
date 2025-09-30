from pg_database import PG_Database

class Benchmark:


    def __init__(self, benchmark="TPCH"):
        self.benchmark = benchmark
        if (self.benchmark.upper() == "TPCH"):
            wk_file = 'data/workload/tpch_custom_20.sql'
            with open(wk_file, 'r') as f:
                content = f.read()
                queries = content.split('\n')
                if queries[-1] == '':
                    queries.pop()
            self.queries =  queries
        elif self.benchmark.upper() == 'CEB':
            wk_file = 'data/workload/ceb_16.sql'
            with open(wk_file, 'r') as f:
                content = f.read()
                queries = content.split('\n')
                if queries[-1] == '':
                    queries.pop()
            self.queries =  queries
        elif self.benchmark.upper() == 'TPCDS':
            wk_file = 'data/workload/tpcds_custom_20.sql'
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
