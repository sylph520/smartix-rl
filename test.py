import psycopg2

actual_runs = 3
db_user = "sclai"
db_name = "indexselection_tpch___1"
db_port = 51204
indexes0 = {'c_custkey': 0, 'c_name': 0, 'c_address': 0, 'c_nationkey': 1, 'c_phone': 0, 'c_acctbal': 0, 'c_mktsegment': 0, 'c_comment': 0, 'l_partkey': 1, 'l_suppkey': 0, 'l_quantity': 1, 'l_extendedprice': 1, 'l_discount': 0, 'l_tax': 0, 'l_returnflag': 0, 'l_linestatus': 0, 'l_shipdate': 0, 'l_commitdate': 1, 'l_receiptdate': 0, 'l_shipinstruct': 0, 'l_shipmode': 0, 'l_comment': 1, 'n_nationkey': 0, 'n_name': 0, 'n_regionkey': 0, 'n_comment': 0, 'o_custkey': 1, 'o_orderstatus': 0, 'o_totalprice': 1, 'o_orderdate': 0, 'o_orderpriority': 0, 'o_clerk': 1, 'o_shippriority': 0, 'o_comment': 0, 'p_partkey': 0, 'p_name': 0, 'p_mfgr': 1, 'p_brand': 1, 'p_type': 0, 'p_size': 0, 'p_container': 0, 'p_retailprice': 0, 'p_comment': 0, 'ps_partkey': 0, 'ps_suppkey': 1, 'ps_availqty': 0, 'ps_supplycost': 1, 'ps_comment': 1, 'r_regionkey': 0, 'r_name': 0, 'r_comment': 0, 's_suppkey': 1, 's_name': 0, 's_address': 0, 's_nationkey': 1, 's_phone': 1, 's_acctbal': 0, 's_comment': 1}
indexes1 = [k for k, v in indexes0.items() if v==1]
indexes = []
for idx in indexes1:
    if idx[0] == 'c':
        tidx = ('customer', idx)
    elif idx[0] == 'l':
        tidx = ('lineitem', idx)
    elif idx[0] == 'o':
        tidx = ('orders', idx)
    elif idx[0] == 'p':
        if idx[1] == '_':
            tidx = ('part', idx)
        else:
            tidx = ('partsupp', idx)
    elif idx[0] == 's':
        tidx = ('supplier', idx)
    indexes.append(tidx)
querys = [
	"SELECT nation.n_name FROM region, nation WHERE nation.n_regionkey = region.r_regionkey AND nation.n_nationkey < 2 AND region.r_comment = 'ly final courts cajole furiously final excuse';",
	"SELECT partsupp.ps_partkey FROM partsupp, supplier WHERE supplier.s_suppkey = partsupp.ps_suppkey AND supplier.s_nationkey = 20 AND supplier.s_suppkey = 2682;",
	"SELECT nation.n_regionkey FROM region, nation WHERE nation.n_regionkey = region.r_regionkey AND nation.n_comment = 'c dependencies. furiously express notornis sleep slyly regular accounts. ideas sleep. depos' AND nation.n_name = 'EGYPT                    ';",
	"SELECT lineitem.l_linenumber FROM partsupp, lineitem WHERE lineitem.l_partkey = partsupp.ps_partkey AND lineitem.l_orderkey < 1296422 AND partsupp.ps_partkey = 35538;",
	"SELECT customer.c_acctbal FROM customer, orders WHERE customer.c_custkey = orders.o_custkey AND orders.o_clerk = 'Clerk#000000107' AND customer.c_address = 'cR5QLdFoi6e2K96SOIuBSTagcn9orFt7,xWx';",
	"SELECT customer.c_acctbal FROM customer, orders WHERE customer.c_custkey = orders.o_custkey AND customer.c_address = 'eg5fr71AHlvZ4II4mNQoO' AND orders.o_comment = 'egular requests integrate quickly ';",
	"SELECT supplier.s_name FROM lineitem, supplier WHERE lineitem.l_suppkey = supplier.s_suppkey AND supplier.s_phone = '26-265-429-8153' AND lineitem.l_shipmode = 'MAIL      ';",
	"SELECT partsupp.ps_suppkey FROM partsupp, lineitem WHERE lineitem.l_partkey = partsupp.ps_partkey AND lineitem.l_orderkey > 868036 AND lineitem.l_extendedprice < 42570.15;",
	"SELECT orders.o_custkey FROM orders, lineitem WHERE lineitem.l_orderkey = orders.o_orderkey AND lineitem.l_extendedprice = 19092.59 AND lineitem.l_suppkey < 7808;",
	"SELECT part.p_retailprice FROM part, partsupp WHERE part.p_partkey = partsupp.ps_partkey AND part.p_name = 'sky burlywood lawn almond tan' AND part.p_brAND = 'BrAND#25  ';",
	"SELECT lineitem.l_partkey FROM partsupp, lineitem WHERE lineitem.l_partkey = partsupp.ps_partkey AND partsupp.ps_partkey = 105677 AND lineitem.l_linestatus = 'O';",
	"SELECT lineitem.l_quantity FROM orders, lineitem WHERE lineitem.l_orderkey = orders.o_orderkey AND lineitem.l_returnflag = 'R' AND lineitem.l_extendedprice > 45771.32;",
	"SELECT lineitem.l_shipmode FROM partsupp, lineitem WHERE lineitem.l_suppkey = partsupp.ps_suppkey AND lineitem.l_quantity > 32.00 AND lineitem.l_partkey < 132057;",
	"SELECT partsupp.ps_suppkey FROM partsupp, supplier WHERE supplier.s_suppkey = partsupp.ps_suppkey AND partsupp.ps_partkey > 36832 AND partsupp.ps_supplycost > 640.38;",
	"SELECT lineitem.l_linestatus FROM part, lineitem WHERE lineitem.l_partkey = part.p_partkey AND lineitem.l_comment = 'ial, regular asymptotes wake quickly regula' AND lineitem.l_shipmode = 'TRUCK     ';",
	"SELECT lineitem.l_quantity FROM partsupp, lineitem WHERE lineitem.l_suppkey = partsupp.ps_suppkey AND lineitem.l_linenumber = 3 AND lineitem.l_shipmode = 'MAIL      ';",
	"SELECT orders.o_totalprice FROM customer, orders WHERE customer.c_custkey = orders.o_custkey AND customer.c_nationkey > 23 AND orders.o_orderpriority = '1-URGENT       ';",
	"SELECT lineitem.l_linestatus FROM lineitem, supplier WHERE lineitem.l_suppkey = supplier.s_suppkey AND lineitem.l_commitdate = date'1998-06-16' AND lineitem.l_extendedprice < 40722.24;",
	"SELECT partsupp.ps_availqty FROM part, partsupp WHERE part.p_partkey = partsupp.ps_partkey AND partsupp.ps_comment = 'en somas are around the furiously final deposits. quickly ironic instructions' AND partsupp.ps_supplycost > 307.11;",
	"SELECT lineitem.l_commitdate FROM partsupp, lineitem WHERE lineitem.l_partkey = partsupp.ps_partkey AND lineitem.l_suppkey > 4878 AND lineitem.l_tax = 0.05;"
]
querys_time = []
querys_cost = []


if __name__ == "__main__":
    # 建立连接
    connection = psycopg2.connect(user=db_user, port=db_port, database=db_name, host='/tmp')
    cursor = connection.cursor()
    
    # 建立索引
    for i in range(len(indexes)):
        cursor.execute(f"create index index{i} on {indexes[i][0]} ({indexes[i][1]})")
    
    # 计算cost
    for query in querys:
        cursor.execute(f"explain (format json) {query}")
        result = cursor.fetchone()
        querys_cost.append(result[0][0]["Plan"]["Total Cost"])
    
    # 计算执行时间
    for query in querys:
        total_time = 0.0
        for i in range(actual_runs):
            cursor.execute(f"explain (analyze, buffers, format json) {query}")
            result = cursor.fetchone()
            total_time += result[0][0]["Plan"]["Actual Total Time"]
        querys_time.append(total_time / actual_runs)
    
    # 删除索引
    for i in range(len(indexes)):
        cursor.execute(f"drop index index{i}")
    
    # 关闭连接
    cursor.close()
    connection.close()
    
    # 输出结果
    print(querys_time)
    print(querys_cost)
    print(sum(querys_time))
    print(sum(querys_cost))