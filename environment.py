from pg_database import PG_Database
import numpy as np
import time
import random

class Environment():
    def __init__(self, workload_path='data/workload/tpch.sql', hypo=True, allow_columns=True, flip=True, no_op=True, window_size = 80):
        # DBMS
        self.db = PG_Database(hypo=hypo)

        # Database
        self.table_columns = self.db.tables
        self.tables = list(self.table_columns.keys())
        self.columns = np.concatenate(list(self.table_columns.values())).tolist()

        # Workload
        self.workload = self.load_workload(workload_path)
        self.workload_iterator = 0
        self.window_size = window_size

        # Environment
        self.allow_columns = allow_columns
        self.flip = flip
        self.no_op = no_op
        if self.allow_columns:
            self.n_features = len(self.columns) * 2
            self.n_actions = len(self.columns) * 2
        else:
            self.n_features = len(self.columns)
            self.n_actions = len(self.columns) * 2
        if self.flip:
            self.n_actions = len(self.columns)
        if self.no_op:
            self.n_actions += 1

        # DEBUG
        self.optimal_indexes = ['c_acctbal', 'p_brand', 'p_container', 'p_size', 'l_shipdate', 'o_orderdate']

    """
        Acessed from outside
        - step(action)
        - reset()
        - close()
    """
    def step(self, action):
        # Apply action
        s_prime, reward = self.apply_transition(action)

        return s_prime, reward, False, dict()
    
    def reset(self):
        # Workload and indexes
        self.db.reset_indexes()        
        self.workload_iterator = 0

        # Window-related
        self.workload_buffer, self.usage_history = self.initialize_window(self.window_size)

        return self.get_state()
    
    def close(self):
        self.db.reset_indexes()
        return self.db.close_connection()

    def debug(self):
        indexes_dict = self.db.get_indexes()
        if indexes_dict['c_acctbal'] == 1: print('c_acctbal')
        if indexes_dict['p_brand'] == 1: print('p_brand')
        if indexes_dict['p_container'] == 1: print('p_container')
        if indexes_dict['p_size'] == 1: print('p_size')
        if indexes_dict['l_shipdate'] == 1: print('l_shipdate')
        if indexes_dict['o_orderdate'] == 1: print('o_orderdate')
        print(sum(indexes_dict.values()))
        print("")

    """
        Reward functions
            compute_reward_weight_columns
            compute_reward_weight_indexes
            compute_reward_index_scan
            compute_reward_avg_cost_window
            compute_reward_cost
            compute_reward_cost_difference
    """

    def compute_reward_weight_columns(self, changed, drop, table, column):
        column_count = self.get_column_count_window()
        count = column_count[self.columns.index(column)]
        
        threshold = 1
        reward = 0
        if drop and count > threshold:
            reward = -count * 2
        elif drop and count <= threshold:
            reward = count
        elif not drop and count <= threshold:
            reward = -count * 2
        elif not drop and count > threshold:
            reward = count

        return reward
    
    def compute_reward_weight_indexes(self, changed, drop, table, column):
        if not changed: 
            print('NOT CHANGED')
            return -1

        index_count = self.get_index_count_window()
        count = index_count[self.columns.index(column)]

        if count > 0: print(column, count)

        reward = 10
        if drop and count > 0:
            reward = -(reward*count) * 4
        if not drop and count > 0:
            reward = (reward*count) * 2
        if not drop and count == 0:
            reward = -(reward*count) * 4

        return reward

    def compute_reward_index_scan(self, changed, drop, table, column):
        # If action was a drop, create hypothetical
        if drop and changed: self.db.create_index(table, column)

        # Check amount of queries the index could be used in the last len(window)
        total_count = 0
        for q in self.workload_buffer:
            total_use =  self.db.get_query_use(q, column)
            if total_use > 0: break

        # Drop hypothetical in case it was a drop
        if drop and changed: self.db.drop_index(table, column)

        reward = 0
        if changed and drop:
            if total_count == 0: # DROPA == 0 - BOM
                reward = 5
            else: # DROPA > 0 - RUIM
                reward = -20
        if changed and not drop:
            if total_count == 0: # CRIA == 0 - RUIM
                reward = -15
            else: # CRIA > 0 - BOM
                reward = 10
        
        return reward

    def compute_reward_avg_cost_window(self):
        cost = sum(self.cost_history[-self.window_size:])
        cost_avg = cost/self.window_size
        reward = (1/cost_avg) * 100000
        return reward
    
    def compute_reward_cost(self, query):
        cost = self.db.get_query_cost(query)
        reward = (1/cost) * 100000
        return reward
    
    def compute_reward_cost_difference(self, column):
        prev_cost = sum(self.cost_history[-self.window_size:])
        curr_cost = sum([self.db.get_query_cost(q) for q in self.workload_buffer])
        cost_diff = prev_cost - curr_cost
        if cost_diff > 0: 
            reward = 10
            print(column, cost_diff)
        if cost_diff < 0: reward = -20
        else: reward = -1

        return reward
    
    def compute_reward_query_use(self, changed, drop, column):
        use = 0
        usage_count = self.get_usage_count_window()
        if usage_count[self.columns.index(column)] > 0:
            use = 1

        if column in self.optimal_indexes:
            print(drop, '\t', column, '\t', use)

        if drop:
            if use == 0:
                if changed:
                    reward = 1
                else:
                    reward = -5
            else:
                reward = -5
        else:
            if use == 0:
                reward = -5
            else:
                if changed:
                    reward = 1
                else:
                    reward = -5

        return reward

    """
        Transition methods
        - apply_transition(action)
        - apply_index_change(action)
        - step_workload()
        - random_step_workload()
    """

    def apply_transition(self, action):
        # Check if NO OP
        if self.no_op and action == self.n_actions - 1:
            state = self.get_state()
            reward = 0
            # print('NOOP')
            return state, reward

        # Apply index change
        if self.flip:
            changed, drop, table, column = self.apply_index_change_flip(action)
        else:
            changed, drop, table, column = self.apply_index_change(action)

        # Execute next_query
        query = self.step_workload()

        # Update usage history
        self.usage_history.append(self.get_usage_count(column))

        # Compute reward
        # if self.reward_function == 1:
        #     # print(1)
        #     reward = self.compute_reward_weight_columns(changed, drop, table, column)
        # elif self.reward_function == 2:
        #     # print(2)
        #     reward = self.compute_reward_weight_indexes(changed, drop, table, column)
        # elif self.reward_function == 3:
        #     # print(3)
        #     reward = self.compute_reward_index_scan(changed, drop, table, column)
        # elif self.reward_function == 4:
        #     # print(4)
        #     reward = self.compute_reward_avg_cost_window()
        # elif self.reward_function == 5:
        #     # print(5)
        #     reward = self.compute_reward_cost(query)
        # elif self.reward_function == 6:
        #     # print(6)
        #     reward = self.compute_reward_cost_difference(column)
        # elif self.reward_function == 7:
        #     # print(7)
        #     reward = self.compute_reward_query_use(changed, drop, column)
        reward = self.compute_reward_query_use(changed, drop, column)

        # Get next state
        s_prime = self.get_state()

        return s_prime, reward
    
    def apply_index_change(self, action):
        indexes = self.db.get_indexes()

        if self.no_op:
            n_actions = self.n_actions - 1
        else:
            n_actions = self.n_actions

        # Check if drop
        drop = False
        if action >= n_actions/2: 
            drop = True
            action = int(action - n_actions/2)
        
        # Get table and column
        column = self.columns[action]
        for table_name in self.tables:
            if column in self.table_columns[table_name]:
                table = table_name

        # Apply change
        if drop:
            if indexes[column] == 0: 
                changed = False
            else:
                self.db.drop_index(table, column)
                changed = True
        else:
            if indexes[column] == 1:
                changed = False
            else: 
                self.db.create_index(table, column)
                changed = True
                
        return changed, drop, table, column

    def apply_index_change_flip(self, action):
        indexes = self.db.get_indexes()

        # Get table and column
        column = self.columns[action]

        for table_name, columns in self.table_columns.items():
            if column in columns:
                table = table_name
                break
        
        # WORKAROUND FOR WEIRD COLUMN THAT DOES NOT WORK CREATING INDEX
        if column == 's_comment': return False, False, table, column

        # Apply change
        if indexes[column] == 1: 
            self.db.drop_index(table, column)
            drop = True
        else:
            self.db.create_index(table, column)
            drop = False
        
        return True, drop, table, column

    def step_workload(self):
        query = self.workload[self.workload_iterator]

        # Execute query
        # start = time.time()
        # self.db.execute(query, verbose=True)
        # end = time.time()
        # elapsed_time = end - start
        
        # Update window
        # self.column_count.append(self.get_column_count(query))
        # self.index_count.append(self.get_index_count(query))
        # self.cost_history.append(self.db.get_query_cost(query))
        self.workload_buffer.append(query)
        self.workload_buffer.pop(0)

        # Manage iterator
        self.workload_iterator += 1
        if self.workload_iterator == len(self.workload):
            self.workload_iterator = 0
        
        return query
    
    def random_step_workload(self):
        query = random.choice(self.workload)

        # Execute query
        # start = time.time()
        # self.db.execute(query)
        # end = time.time()
        # elapsed_time = end - start
        
        # Update window
        # self.column_count.append(self.get_column_count(query))
        # self.index_count.append(self.get_index_count(query))
        # self.cost_history.append(self.db.get_query_cost(query))
        self.workload_buffer.append(query)
        self.workload_buffer.pop(0)

        return query

    """
        Count updates
        - update_column_count(query)
        - get_column_count_window()
    """

    # def get_column_count(self, query):
    #     column_count = [0] * len(self.columns)
    #     where_columns = self.get_where_columns(query)
    #     for column in where_columns:
    #         column_count[self.columns.index(column)] = 1
    #     return column_count
    
    # def get_column_count_window(self):
    #     total_count = [0] * len(self.columns)
    #     for count in self.column_count[-self.window_size:]:
    #         total_count = [sum(col) for col in zip(total_count, count)]
    #     return total_count

    # def get_index_count(self, query):
    #     index_count = [0] * len(self.columns)
    #     current_indexes = self.db.get_indexes()
    #     where_columns = self.get_where_columns(query)
    #     for column in where_columns:
    #         for table in self.tables:
    #             if column in self.table_columns[table]:
    #                 if current_indexes[column] == 1: create = False
    #                 else: create = True
    #                 if create: self.db.create_index(table, column)
    #                 count = self.db.get_query_use(query, column)
    #                 index_count[self.columns.index(column)] = count
    #                 if create: self.db.drop_index(table, column)
    #     return index_count
    
    # def get_index_count_window(self):
    #     total_count = [0] * len(self.columns)
    #     for count in self.index_count[-self.window_size:]:
    #         total_count = [sum(col) for col in zip(total_count, count)]
    #     return total_count

    def get_usage_count(self, column):
        usage_count = np.zeros(len(self.columns))
        for query in self.workload_buffer:
            count = self.db.get_query_use(query, column)
            usage_count[self.columns.index(column)] = count
            if count > 0: break
        return usage_count.tolist()

    def get_usage_count_window(self):
        total_count = np.zeros(len(self.columns))
        for count in self.usage_history[-self.window_size:]:
            total_count = [sum(col) for col in zip(total_count, count)]
        return total_count

    def get_where_columns(self, query):
        columns = list()
        select_split = query.split("SELECT")
        for select in select_split:
            if select != '':
                if "WHERE" in str(select).upper():
                    where = select.split("WHERE")[1]
                    avoid = ['GROUP BY', 'ORDER BY', 'LIMIT']
                    for item in avoid:
                        if item in where:
                            where = where.split(item)[0]
                            break
                    for column in self.columns:
                        if str(column).upper() in str(where).upper():
                            columns.append(column)
        return list(set(columns))
    
    """
        Initialization methods
    """

    def initialize_window(self, window_size):
        # column_count = list()
        # index_count = list()
        # cost_history = list()
        workload_buffer = list()
        usage_history = list()

        for query in random.choices(self.workload, k=window_size):
            # cost_history.append(self.db.get_query_cost(query))
            # column_count.append(self.get_column_count(query))
            # index_count.append(self.get_index_count(query))
            workload_buffer.append(query)
            usage_history.append(np.zeros(len(self.columns)).tolist())  # Blank init

        return workload_buffer, usage_history
    
    def get_state(self):
        if self.allow_columns:
            indexes = np.array(list(self.db.get_indexes().values()), dtype=int)
            # current_count = np.array(self.get_column_count_window())
            # current_count = np.array(self.get_index_count_window())
            current_count = np.array(self.get_usage_count_window(), dtype=int)
            state = np.concatenate((indexes, current_count))
        else:
            indexes = np.array(list(self.db.get_indexes().values()), dtype=int)
            state = indexes
        return state
    
    def load_workload(self, path):
        with open(path, 'r') as f:
            data = f.read()
        workload = data.split('\n')
        return workload


if __name__ == "__main__":
    from pprint import pprint
    
    env = Environment(hypo=True)
    env.reset()

    buffer = list()
    window = 22
    iterator = 0
    while True:
        buffer.append(env.workload[iterator])
        iterator += 1
        if iterator == len(env.workload):
            iterator = 0
        if len(buffer) == window: break
    print('Buffer', len(buffer))

    while True:
        new_query = env.workload[iterator]
        buffer.append(new_query)
        buffer.pop(0)
        iterator += 1
        if iterator == len(env.workload): iterator = 0
        for col in env.optimal_indexes:
            total_use = 0
            for q in buffer:
                total_use = env.db.get_query_use(q, col)
                if total_use > 0: 
                    break
            if total_use == 0: print('MEU DEUSSSSSSSSSSSS')
            # print(total_use, col, len(buffer))
        # print("")

    env.close()
